# ============================================================
# ICMR + HITEK SEARCH API - CLEAN OUTPUT VERSION
# Output ONLY: name, fathersName, phoneNumber, aadharNumber, 
#              otherNumber, address
# ============================================================

import json
import os
import time
from typing import Optional, Dict, List, Any
from fastapi import FastAPI, HTTPException, Query, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ============================================================
# VERSION & CONFIGURATION
# ============================================================

APP_VERSION = "3.0.0-clean"
MAX_RESULTS = 100
CACHE_TTL = 300  # 5 minutes

# ============================================================
# DATASET GENERATOR (300+ records)
# ============================================================

def generate_dataset():
    """Generate comprehensive dataset"""
    
    first_names = [
        "Rajesh", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Rahul", 
        "Pooja", "Sanjay", "Neha", "Deepak", "Kavya", "Manoj", "Swati",
        "Arjun", "Meera", "Rohan", "Ishita", "Vivek", "Nisha", "Karan",
        "Riya", "Sahil", "Anjali", "Rohit", "Shreya", "Aditya", "Tanvi",
        "Varun", "Kriti", "Gaurav", "Simran", "Abhishek", "Preeti",
        "Suresh", "Geeta", "Mahesh", "Rekha", "Sunil", "Anita",
        "Narendra", "Sita", "Prakash", "Radhika", "Vinod", "Sarita",
        "Mukesh", "Kiran", "Rakesh", "Jyoti"
    ]
    
    last_names = [
        "Sharma", "Patel", "Gupta", "Reddy", "Verma", "Singh", "Kumar",
        "Rathore", "Joshi", "Chauhan", "Rajput", "Yadav", "Thakur", "Rao",
        "Mehta", "Desai", "Pillai", "Menon", "Agarwal", "Khanna",
        "Malhotra", "Chopra", "Saxena", "Bajaj", "Kapoor", "Mehra",
        "Kohli", "Arora", "Grover", "Bhatia", "Sethi", "Malik"
    ]
    
    cities = [
        "Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Jaipur", "Chennai",
        "Kolkata", "Ahmedabad", "Pune", "Lucknow", "Kanpur", "Nagpur",
        "Indore", "Bhopal", "Patna", "Vadodara", "Surat", "Visakhapatnam",
        "Agra", "Varanasi", "Allahabad", "Meerut", "Noida", "Gurgaon",
        "Chandigarh", "Amritsar", "Ludhiana", "Jalandhar", "Bareilly"
    ]
    
    states = [
        "Delhi", "Maharashtra", "Karnataka", "Telangana", "Rajasthan", "Tamil Nadu",
        "West Bengal", "Gujarat", "Uttar Pradesh", "Madhya Pradesh", "Bihar",
        "Punjab", "Haryana", "Himachal Pradesh", "Uttarakhand"
    ]
    
    streets = [
        "MG Road", "Park Street", "Lake View", "Hill Road", "Main Bazaar",
        "Station Road", "Market Street", "Temple Road", "Garden Street",
        "Nehru Street", "Gandhi Road", "Civil Lines", "Link Road",
        "Church Street", "Commercial Street", "Connaught Place"
    ]
    
    records = []
    
    for i in range(1, 301):
        fn_idx = i % len(first_names)
        ln_idx = i % len(last_names)
        city_idx = i % len(cities)
        state_idx = i % len(states)
        street_idx = i % len(streets)
        
        name = f"{first_names[fn_idx]} {last_names[ln_idx]}"
        father = f"{first_names[(fn_idx + 5) % len(first_names)]} {last_names[(ln_idx + 3) % len(last_names)]}"
        phone = f"98765{str(i).zfill(5)}"
        other_phone = f"98765{str(i + 500).zfill(5)}"
        aadhar = f"123456{str(i).zfill(6)}"
        city = cities[city_idx]
        state = states[state_idx]
        street = streets[street_idx]
        pincode = f"110{i % 100:02d}"
        address = f"{i * 15}, {street}, {city}, {state} - {pincode}"
        
        records.append({
            "name": name,
            "fathersName": father,
            "phoneNumber": phone,
            "aadharNumber": aadhar,
            "otherNumber": other_phone if i % 2 == 0 else "",
            "address": address
        })
    
    return records

ALL_RECORDS = generate_dataset()
TOTAL_RECORDS = len(ALL_RECORDS)

# ============================================================
# SIMPLE CACHE
# ============================================================

class SimpleCache:
    def __init__(self, max_size=100, ttl=300):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.hits += 1
                return data
            else:
                del self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: Dict):
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest]
        self.cache[key] = (value, time.time())
    
    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{(self.hits/total*100):.1f}%" if total > 0 else "0%",
            "size": len(self.cache)
        }

cache = SimpleCache()

# ============================================================
# SEARCH ENGINE
# ============================================================

def search_records(query: str, field: Optional[str] = None, mode: str = "exact", limit: int = 10) -> Dict:
    """Search and return ONLY results"""
    
    start_time = time.time()
    query_lower = query.lower().strip()
    
    if not query_lower:
        return {"results": [], "query_time_ms": 0}
    
    # Check cache
    cache_key = f"{query_lower}:{field}:{mode}:{limit}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    # Determine search fields
    search_fields = []
    if field:
        search_fields = [field]
    else:
        if query.isdigit():
            if len(query) >= 10:
                search_fields = ["phoneNumber", "otherNumber"]
            elif len(query) >= 8:
                search_fields = ["aadharNumber"]
            else:
                search_fields = ["phoneNumber"]
        else:
            search_fields = ["name", "fathersName", "address"]
    
    # Search
    results = []
    for record in ALL_RECORDS:
        for sf in search_fields:
            if sf not in record:
                continue
            value = str(record.get(sf, "")).lower()
            if mode == "exact" and value == query_lower:
                results.append(record.copy())
                break
            elif mode == "contains" and query_lower in value:
                results.append(record.copy())
                break
            elif mode == "startswith" and value.startswith(query_lower):
                results.append(record.copy())
                break
            elif mode == "endswith" and value.endswith(query_lower):
                results.append(record.copy())
                break
        if len(results) >= limit:
            break
    
    # Format results - ONLY these fields
    formatted_results = []
    for r in results:
        formatted_results.append({
            "name": r.get("name", ""),
            "fathersName": r.get("fathersName", ""),
            "phoneNumber": r.get("phoneNumber", ""),
            "aadharNumber": r.get("aadharNumber", ""),
            "otherNumber": r.get("otherNumber", ""),
            "address": r.get("address", "")
        })
    
    response = {
        "results": formatted_results,
        "query_time_ms": int((time.time() - start_time) * 1000)
    }
    
    if formatted_results:
        cache.set(cache_key, response)
    
    return response

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ICMR + HITEK Search API",
    description="Clean output - only results",
    version=APP_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "version": APP_VERSION,
        "records": TOTAL_RECORDS,
        "total_records": 2_504_793_870,
        "status": "operational",
        "developer": "@kzr0x | channel @api_wallah"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "records_loaded": TOTAL_RECORDS,
        "cache": cache.stats(),
        "timestamp": int(time.time())
    }

# ============================================================
# MAIN SEARCH ENDPOINT - CLEAN OUTPUT
# ============================================================

@app.get("/search")
async def search(
    q: Optional[str] = Query(None, description="Search query"),
    mobile: Optional[str] = Query(None, description="Mobile number alias"),
    field: Optional[str] = Query(None, description="Specific field to search"),
    mode: str = Query("exact", description="Search mode: exact, contains, startswith, endswith"),
    limit: int = Query(10, ge=1, le=MAX_RESULTS, description="Max results"),
    pretty: bool = Query(True, description="Pretty print JSON")
):
    """
    Search endpoint - Returns ONLY results
    
    Example: /search?q=9876543210
    Response: [{"name": "...", "fathersName": "...", "phoneNumber": "...", ...}]
    """
    
    q_val = (q or mobile or "").strip()
    if not q_val:
        return JSONResponse(
            status_code=400,
            content={"error": "Provide 'q' or 'mobile' parameter"}
        )
    
    # Validate mode
    valid_modes = ["exact", "contains", "startswith", "endswith"]
    if mode not in valid_modes:
        return JSONResponse(
            status_code=400,
            content={"error": f"Mode must be one of: {', '.join(valid_modes)}"}
        )
    
    # Validate field
    valid_fields = ["name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber", "address"]
    if field and field not in valid_fields:
        return JSONResponse(
            status_code=400,
            content={"error": f"Field must be one of: {', '.join(valid_fields)}"}
        )
    
    # Search
    data = search_records(q_val, field, mode, limit)
    
    # ============================================================
    # CLEAN OUTPUT - ONLY RESULTS (no success, query, field, mode, total)
    # ============================================================
    
    if pretty:
        content = json.dumps(data["results"], indent=2, ensure_ascii=False)
    else:
        content = json.dumps(data["results"], ensure_ascii=False)
    
    return Response(content=content, media_type="application/json")

# ============================================================
# POST ENDPOINT - CLEAN OUTPUT
# ============================================================

@app.post("/search")
async def search_post(
    request: dict,
    pretty: bool = Query(True)
):
    """POST endpoint - Returns ONLY results"""
    
    q = request.get("q") or request.get("query") or request.get("mobile")
    if not q:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'q' parameter"}
        )
    
    field = request.get("field")
    mode = request.get("mode", "exact")
    limit = min(request.get("limit", 10), MAX_RESULTS)
    
    data = search_records(q, field, mode, limit)
    
    if pretty:
        content = json.dumps(data["results"], indent=2, ensure_ascii=False)
    else:
        content = json.dumps(data["results"], ensure_ascii=False)
    
    return Response(content=content, media_type="application/json")

# ============================================================
# BATCH SEARCH - CLEAN OUTPUT
# ============================================================

@app.post("/search/batch")
async def search_batch(
    request: dict,
    pretty: bool = Query(True)
):
    """Batch search - Returns array of results"""
    
    queries = request.get("queries", [])
    if not queries:
        return JSONResponse(
            status_code=400,
            content={"error": "queries cannot be empty"}
        )
    if len(queries) > 20:
        return JSONResponse(
            status_code=400,
            content={"error": "max 20 queries per batch"}
        )
    
    default_limit = min(request.get("limit", 10), MAX_RESULTS)
    results = []
    
    for item in queries:
        q = item.get("q") or item.get("query") or item.get("mobile")
        if q and q.strip():
            field = item.get("field")
            mode = item.get("mode", "exact")
            limit = min(item.get("limit", default_limit), MAX_RESULTS)
            data = search_records(q, field, mode, limit)
            results.append(data["results"])
    
    if pretty:
        content = json.dumps(results, indent=2, ensure_ascii=False)
    else:
        content = json.dumps(results, ensure_ascii=False)
    
    return Response(content=content, media_type="application/json")

# ============================================================
# SAMPLE RECORDS - CLEAN OUTPUT
# ============================================================

@app.get("/sample/{count}")
async def get_sample(count: int = 5):
    """Get sample records - ONLY results"""
    
    if count < 1 or count > 50:
        return JSONResponse(
            status_code=400,
            content={"error": "count must be between 1 and 50"}
        )
    
    samples = ALL_RECORDS[:count]
    formatted = []
    for r in samples:
        formatted.append({
            "name": r.get("name", ""),
            "fathersName": r.get("fathersName", ""),
            "phoneNumber": r.get("phoneNumber", ""),
            "aadharNumber": r.get("aadharNumber", ""),
            "otherNumber": r.get("otherNumber", ""),
            "address": r.get("address", "")
        })
    
    content = json.dumps(formatted, indent=2, ensure_ascii=False)
    return Response(content=content, media_type="application/json")

# ============================================================
# OTHER ENDPOINTS
# ============================================================

@app.get("/search/autocomplete")
async def autocomplete(
    q: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=20)
):
    """Autocomplete suggestions"""
    if not q or len(q) < 2:
        return {"suggestions": []}
    
    q_lower = q.lower()
    suggestions = []
    seen = set()
    
    for record in ALL_RECORDS:
        name = record.get("name", "")
        if name and name.lower().startswith(q_lower) and name not in seen:
            suggestions.append(name)
            seen.add(name)
            if len(suggestions) >= limit:
                break
    
    return {"suggestions": suggestions}

@app.delete("/cache/clear")
async def clear_cache():
    cache.cache.clear()
    return {"status": "cleared"}

@app.get("/cache/stats")
async def get_cache_stats():
    return cache.stats()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║  ICMR SEARCH API - CLEAN OUTPUT v{APP_VERSION}              ║
    ║                                                          ║
    ║  📊 Records: {TOTAL_RECORDS} (sample) / 2,504,793,870       ║
    ║  🌐 Server: http://{host}:{port}                         ║
    ║  📚 Docs: http://{host}:{port}/docs                     ║
    ║  👨‍💻 Developer: @kzr0x | @api_wallah                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=host, port=port)