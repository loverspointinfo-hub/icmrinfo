import asyncio
import json
import os
import threading
import hashlib
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

# Force early initialization for Vercel
import duckdb
import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

# ── Vercel-specific fixes ──────────────────────────────────────────────────
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("DUCKDB_HOME", "/tmp")
os.environ.setdefault("ICMR_PARALLEL", "1")
os.environ.setdefault("ICMR_THREADS_PER_CONN", "1")

# ── Config ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
HF_INDEX_BASE = os.environ.get(
    "ICMR_HF_INDEX_BASE",
    "https://huggingface.co/datasets/Kzr0xx/icrm-hitek-full-db-mixed/resolve/main",
).rstrip("/")
INDEX_SOURCE = os.environ.get("ICMR_INDEX_SOURCE", "remote").lower()
PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "1"))
THREADS_PER_CONN = int(os.environ.get("ICMR_THREADS_PER_CONN", "1"))
DUPLICATE_CAP = 2
CACHE_SIZE = 100
CACHE_TTL = 300

# Primary display fields (only these will be shown)
DISPLAY_FIELDS = ["name", "fathersName", "aadharNumber", "otherNumber", "address"]

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", 
    "otherNumber", "address", "district", "pincode", 
    "state", "town", "source"
]

PRIMARY_FIELDS = ["name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber", "address"]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

REMOTE_INDEXES = {
    "phone": [f"{HF_INDEX_BASE}/idx_phone.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_INDEX_BASE}/idx_aadhar.{i}.parquet" for i in range(7)],
}

# ── Cache Implementation ──────────────────────────────────────────────────
class QueryCache:
    """Simple LRU cache with TTL for query results"""
    
    def __init__(self, max_size: int = CACHE_SIZE, ttl: int = CACHE_TTL):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def _get_cache_key(self, query: str, field: str, mode: str, limit: int) -> str:
        key_string = f"{query}|{field}|{mode}|{limit}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, field: str, mode: str, limit: int) -> Optional[Dict]:
        key = self._get_cache_key(query, field, mode, limit)
        
        with self.lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return result
                else:
                    del self.cache[key]
            self.misses += 1
        return None
    
    def set(self, query: str, field: str, mode: str, limit: int, result: Dict):
        key = self._get_cache_key(query, field, mode, limit)
        
        with self.lock:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = (result, time.time())
            self.cache.move_to_end(key)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict:
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.1f}%"
            }

# Initialize cache
query_cache = QueryCache()

# ── Lazy DuckDB Connection ──────────────────────────────────────────────────
_conn = None
_conn_lock = threading.Lock()
_initialized = False

def _init_duckdb():
    """Initialize DuckDB with lazy loading - only when needed"""
    global _conn, _initialized
    
    if _initialized and _conn is not None:
        return _conn
    
    with _conn_lock:
        if _initialized and _conn is not None:
            return _conn
        
        try:
            print("Initializing DuckDB connection...")
            con = duckdb.connect(database=':memory:')
            
            con.execute("SET home_directory='/tmp'")
            con.execute("SET extension_directory='/tmp/duckdb_extensions'")
            con.execute("SET max_memory='256MB'")
            con.execute("SET temp_directory='/tmp'")
            
            try:
                con.execute("INSTALL parquet; LOAD parquet;")
                con.execute("INSTALL httpfs; LOAD httpfs;")
            except Exception as e:
                print(f"Extension installation warning: {e}")
            
            for kind, urls in REMOTE_INDEXES.items():
                view = f"people_{kind}"
                try:
                    lst = ", ".join(f"'{u}'" for u in urls)
                    con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet([{lst}])")
                except Exception as e:
                    print(f"Failed to create view {view}: {e}")
            
            con.execute(f"SET threads = {THREADS_PER_CONN}")
            _create_sample_data(con)
            
            _conn = con
            _initialized = True
            print("DuckDB initialized successfully")
            return con
            
        except Exception as e:
            print(f"Error initializing DuckDB: {e}")
            return _get_minimal_connection()


def _get_minimal_connection():
    """Get minimal DuckDB connection for sample data only"""
    try:
        con = duckdb.connect(database=':memory:')
        con.execute("SET home_directory='/tmp'")
        con.execute("SET extension_directory='/tmp/duckdb_extensions'")
        return con
    except:
        return None


def _create_sample_data(con):
    """Create sample records only if database is empty"""
    if con is None:
        return
    
    try:
        result = con.execute("SELECT COUNT(*) FROM people_phone").fetchone()
        if result and result[0] > 0:
            return
    except:
        pass
    
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS people_phone (
                name VARCHAR,
                fathersName VARCHAR,
                phoneNumber VARCHAR,
                aadharNumber VARCHAR,
                otherNumber VARCHAR,
                address VARCHAR,
                district VARCHAR,
                pincode VARCHAR,
                state VARCHAR,
                town VARCHAR,
                source VARCHAR
            )
        """)
        
        sample_names = ["Rajesh Kumar", "Priya Sharma", "Amit Patel", "Sneha Reddy", 
                       "Vikram Singh", "Anjali Gupta", "Ravi Verma", "Kavya Nair",
                       "Suresh Menon", "Meera Iyer", "Arjun Rao", "Divya Shetty",
                       "Manoj Pillai", "Pooja Jain", "Rahul Khanna", "Neha Mishra"]
        
        sample_fathers = ["Ramesh Kumar", "Sita Sharma", "Mohan Patel", "Geeta Reddy",
                         "Shyam Singh", "Radha Gupta", "Gopal Verma", "Lakshmi Nair"]
        
        sample_towns = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata"]
        sample_states = ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu"]
        
        for i in range(50):
            name = sample_names[i % len(sample_names)]
            father = sample_fathers[i % len(sample_fathers)]
            phone = f"9{i:08d}" if len(str(i)) > 4 else f"98{i:07d}"
            aadhar = f"{i+1:012d}"
            other = f"8{i:08d}" if i % 3 == 0 else ""
            town = sample_towns[i % len(sample_towns)]
            
            try:
                con.execute("""
                    INSERT INTO people_phone VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    name,
                    father,
                    phone,
                    aadhar,
                    other,
                    f"#{i+1}, {town} Main Road, Near {town} Station",
                    town,
                    f"{100000 + i}",
                    sample_states[i % len(sample_states)],
                    town,
                    "sample_data"
                ])
            except:
                pass
        
        print(f"Created sample records")
    except Exception as e:
        print(f"Error creating sample data: {e}")


def _get_conn():
    """Get or create DuckDB connection"""
    if _conn is None:
        return _init_duckdb()
    return _conn


# ── Search Functions ──────────────────────────────────────────────────────
def _detect_search_field(query: str) -> Tuple[str, str]:
    """Auto-detect best field and mode for query"""
    q = query.strip()
    
    digits = ''.join(filter(str.isdigit, q))
    if digits:
        if len(digits) == 10 and q.replace("+", "").replace("-", "").replace(" ", "").isdigit():
            return ("phoneNumber", "exact")
        elif len(digits) == 12:
            return ("aadharNumber", "exact")
        elif len(digits) >= 6:
            return ("phoneNumber", "contains")
    
    address_keywords = ["road", "street", "lane", "avenue", "colony", "society", "building", "#"]
    if any(keyword in q.lower() for keyword in address_keywords):
        return ("address", "contains")
    
    if q.replace(" ", "").isalpha() and len(q) >= 3:
        return ("name", "contains")
    
    return ("name", "contains")


def _filter_display_fields(record: dict) -> dict:
    """Filter record to only show display fields"""
    filtered = {}
    for field in DISPLAY_FIELDS:
        if field in record:
            filtered[field] = record[field]
    return filtered


def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    """Run search with caching and timing"""
    start_time = time.time()
    
    if field not in SEARCH_FIELDS:
        return {
            "field": field, 
            "value": value, 
            "mode": mode, 
            "count": 0, 
            "results": [], 
            "error": "Invalid field",
            "response_time_ms": 0
        }
    
    # Check cache
    cached = query_cache.get(value, field, mode, limit)
    if cached:
        cached["from_cache"] = True
        cached["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
        return cached
    
    con = _get_conn()
    if con is None:
        return {
            "field": field, 
            "value": value, 
            "mode": mode, 
            "count": 0, 
            "results": [], 
            "error": "Database unavailable",
            "response_time_ms": round((time.time() - start_time) * 1000, 2)
        }
    
    try:
        v = value.replace("'", "''")
        
        if mode == "exact":
            sql = f"SELECT * FROM people_phone WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 10}"
        elif mode == "contains":
            v2 = value.replace("%", r"\%").replace("_", r"\_")
            sql = f"SELECT * FROM people_phone WHERE {field} ILIKE '%{v2}%' ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 10}"
        elif mode == "startswith":
            v2 = value.replace("%", r"\%").replace("_", r"\_")
            sql = f"SELECT * FROM people_phone WHERE {field} ILIKE '{v2}%' ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 10}"
        elif mode == "endswith":
            v2 = value.replace("%", r"\%").replace("_", r"\_")
            sql = f"SELECT * FROM people_phone WHERE {field} ILIKE '%{v2}' ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 10}"
        else:
            return {
                "field": field, 
                "value": value, 
                "mode": mode, 
                "count": 0, 
                "results": [], 
                "error": "Invalid mode",
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            }
        
        rows = con.execute(sql).fetchall()
        cols = [d[0] for d in con.description] if con.description else []
        
        # Filter to only display fields
        filtered_results = []
        for r in rows:
            full_record = dict(zip(cols, r))
            filtered_record = _filter_display_fields(full_record)
            filtered_results.append(filtered_record)
        
        result = {
            "field": field,
            "value": value,
            "mode": mode,
            "count": len(filtered_results),
            "results": filtered_results[:limit],
            "from_cache": False,
            "response_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        query_cache.set(value, field, mode, limit, result)
        return result
        
    except Exception as e:
        return {
            "field": field,
            "value": value,
            "mode": mode,
            "count": 0,
            "results": [],
            "error": str(e),
            "from_cache": False,
            "response_time_ms": round((time.time() - start_time) * 1000, 2)
        }


def _unified_search(q: str, limit: int = 10) -> dict:
    """Unified search with auto-detection and timing"""
    start_time = time.time()
    q = q.strip()
    
    if not q:
        return {
            "query": q, 
            "count": 0, 
            "results": [],
            "response_time_ms": 0
        }
    
    field, mode = _detect_search_field(q)
    results = []
    searched = []
    
    digits = ''.join(filter(str.isdigit, q))
    if digits:
        r = _run_field_search("phoneNumber", q, "exact", limit)
        if r.get("count", 0) > 0:
            results.extend(r["results"])
            searched.append("phoneNumber_exact")
        
        if not results:
            r = _run_field_search("aadharNumber", q, "exact", limit)
            if r.get("count", 0) > 0:
                results.extend(r["results"])
                searched.append("aadharNumber_exact")
    
    if not results:
        r = _run_field_search(field, q, mode, limit)
        if r.get("count", 0) > 0:
            results.extend(r["results"])
            searched.append(f"{field}_{mode}")
    
    # Deduplicate
    seen = set()
    unique_results = []
    for r in results:
        key = _person_key(r)
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    
    return {
        "query": q,
        "detected_field": field,
        "detected_mode": mode,
        "searched_fields": searched,
        "count": len(unique_results),
        "results": unique_results[:limit],
        "from_cache": False,
        "response_time_ms": round((time.time() - start_time) * 1000, 2)
    }


def _person_key(row: dict) -> tuple:
    ph = (row.get("phoneNumber") or "").strip()
    ad = (row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (row.get("name") or "").strip(), (row.get("fathersName") or "").strip()


# ── FastAPI App ──────────────────────────────────────────────────────────
fastapi_app = FastAPI(
    title="ICMR + HITEK Search API",
    description="Search API for 2.5 billion records - Vercel Optimized",
    version="2.0.0"
)

class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10


@fastapi_app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "version": "2.0.0",
        "status": "running",
        "records": "2,504,793,870",
        "sample_records": "50+",
        "display_fields": DISPLAY_FIELDS,
        "features": [
            "Auto field detection",
            "Multiple search modes",
            "Query caching",
            "Deduplication",
            "Fast responses with timing"
        ],
        "docs": "/docs",
        "developer": "@kzr0x"
    }


@fastapi_app.get("/health")
def health():
    cache_stats = query_cache.get_stats()
    return {
        "status": "ok",
        "cache": cache_stats,
        "database_ready": _conn is not None,
        "sample_records": "50+"
    }


@fastapi_app.get("/search")
async def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("auto"),
    limit: int = Query(10, ge=1, le=100),
    pretty: bool = Query(True)
):
    start_time = time.time()
    q_val = (q or mobile or "").strip()
    
    if not q_val:
        raise HTTPException(422, "Provide q or mobile parameter")
    
    loop = asyncio.get_running_loop()
    
    if field:
        if field not in SEARCH_FIELDS:
            raise HTTPException(400, f"Field must be one of: {', '.join(SEARCH_FIELDS)}")
        
        if mode not in ["exact", "contains", "startswith", "endswith"]:
            raise HTTPException(400, "Mode must be: exact, contains, startswith, or endswith")
        
        data = await loop.run_in_executor(None, _run_field_search, field, q_val, mode, limit)
    else:
        if mode == "auto":
            data = await loop.run_in_executor(None, _unified_search, q_val, limit)
        else:
            field, _ = _detect_search_field(q_val)
            data = await loop.run_in_executor(None, _run_field_search, field, q_val, mode, limit)
    
    # Calculate total response time
    total_time = round((time.time() - start_time) * 1000, 2)
    
    # Build response with only footer metadata
    result = {
        "success": data.get("count", 0) > 0,
        "query": q_val,
        "total": data.get("count", 0),
        "from_cache": data.get("from_cache", False),
        "response_time_ms": total_time,
        "search_time_ms": data.get("response_time_ms", 0),
        "results": data.get("results", [])
    }
    
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@fastapi_app.get("/cache/stats")
def cache_stats():
    return {"cache": query_cache.get_stats()}


@fastapi_app.post("/cache/clear")
def clear_cache():
    query_cache.clear()
    return {"message": "Cache cleared"}


# ── Gradio UI ──────────────────────────────────────────────────────────────
def search_ui(query: str, limit: int, mode: str):
    """Clean UI with only display fields and footer metadata"""
    if not query or not query.strip():
        return "⚠️ Please enter a search query"
    
    q = query.strip()
    start_time = time.time()
    
    try:
        if mode == "Auto":
            data = _unified_search(q, int(limit))
        else:
            field, _ = _detect_search_field(q)
            data = _run_field_search(field, q, mode.lower(), int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"
    
    total_time = round((time.time() - start_time) * 1000, 2)
    search_time = data.get("response_time_ms", 0)
    
    count = data.get("count", 0)
    results = data.get("results", [])
    from_cache = data.get("from_cache", False)
    
    if not results:
        return f"""❌ **No results found** for `{q}`

💡 **Tips:**
- Try using "contains" mode for partial matches
- Use "startswith" for prefix searches
- For phone numbers, use "exact" mode
- Try searching by name or address

---
📊 **Response Metadata:**
- Success: false
- Query: {q}
- Total: 0
- From Cache: {from_cache}
- Response Time: {total_time}ms
- Search Time: {search_time}ms"""
    
    # Build clean results with only display fields
    parts = []
    for i, row in enumerate(results, 1):
        lines = []
        for field in DISPLAY_FIELDS:
            val = row.get(field, "")
            if val:
                # Format field names nicely
                display_name = field.replace("fathersName", "Father's Name")
                display_name = display_name.replace("aadharNumber", "Aadhaar Number")
                display_name = display_name.replace("otherNumber", "Other Number")
                lines.append(f"**{display_name}:** {val}")
        
        # If no fields have values, show message
        if not lines:
            lines.append("*(No data available)*")
        
        parts.append("### Result " + str(i) + "\n" + "\n".join(lines))
    
    results_text = "\n\n---\n\n".join(parts)
    
    # Footer with response metadata only
    footer = f"""
---
📊 **Response Metadata:**
- Success: {count > 0}
- Query: {q}
- Total: {count}
- From Cache: {from_cache}
- Response Time: {total_time}ms
- Search Time: {search_time}ms"""
    
    return results_text + footer


def build_ui():
    with gr.Blocks(
        title="ICMR Search API",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0; }
        .subtitle { text-align: center; color: #666; margin-top: 0; }
        .footer { text-align: center; color: #888; margin-top: 20px; }
        .result-card {
            background: #f5f5f5;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
        }
        """
    ) as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search", elem_classes="main-title")
        gr.Markdown("Search **2.5 billion records** — Clean results with response metadata", elem_classes="subtitle")
        
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Name, Phone, Aadhaar, or Address...",
                    lines=1,
                )
            with gr.Column(scale=1):
                limit_slider = gr.Slider(
                    minimum=1, maximum=20, value=10, step=1,
                    label="Max Results",
                )
        
        with gr.Row():
            with gr.Column(scale=2):
                mode_dropdown = gr.Dropdown(
                    choices=["Auto", "Exact", "Contains", "Startswith", "Endswith"],
                    value="Auto",
                    label="Search Mode",
                )
        
        with gr.Row():
            search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        
        output = gr.Markdown(label="Results")
        
        search_btn.click(
            fn=search_ui,
            inputs=[query_input, limit_slider, mode_dropdown],
            outputs=output,
        )
        query_input.submit(
            fn=search_ui,
            inputs=[query_input, limit_slider, mode_dropdown],
            outputs=output,
        )
        
        gr.Markdown("---")
        with gr.Accordion("📡 API Information", open=False):
            gr.Markdown("""
**Endpoints** (via FastAPI):
- `GET /search?q=<query>` — Search with auto-detection
- `GET /search?q=<query>&mode=contains` — Search with specific mode
- `GET /search?field=name&q=<name>&mode=startswith` — Field-specific search
- `GET /health` — Health check with cache stats
- `GET /cache/stats` — Cache statistics
- `POST /cache/clear` — Clear query cache
- `GET /docs` — Swagger UI documentation

**Display Fields:**
- `name` — Full name
- `fathersName` — Father's name
- `aadharNumber` — Aadhaar number (12 digits)
- `otherNumber` — Other contact number
- `address` — Full address

**Search Modes:**
- `exact` — Exact match (fastest)
- `contains` — Contains substring (flexible)
- `startswith` — Starts with prefix
- `endswith` — Ends with suffix

**Response Metadata (footer only):**
- `success` — Whether results were found
- `query` — Your search query
- `total` — Number of results
- `from_cache` — Whether response was cached
- `response_time_ms` — Total API response time
- `search_time_ms` — Actual search execution time

**Source:** [HF Dataset](https://huggingface.co/datasets/Kzr0xx/icrm-hitek-full-db-mixed)
            """)
        
        gr.Markdown(
            "---\n"
            "<div class='footer'>"
            "👨‍💻 **Developer:** @kzr0x  |  📢 **Channel:** @api_wallah  |  "
            "⚡ **v2.0** — Clean results with metadata footer"
            "</div>",
            elem_classes="footer"
        )
    
    return demo


# ── Mount App ──────────────────────────────────────────────────────────────
demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# ── Vercel Handler ─────────────────────────────────────────────────────────
handler = app