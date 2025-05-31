from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from typing import Dict, Optional
import httpx
import logging
from pydantic import BaseModel, HttpUrl

# Enum location (tự sinh docs + validate)
class ProxyLocation(str, Enum):
    korea = "korea"
    germany1 = "germany1"
    vietnam = "vietnam"
    hongkong = "hongkong"
    germany2 = "germany2"

# Danh sách proxy
PROXY_LIST = {
    ProxyLocation.korea: {
        "http": "http://123.141.185.139:5031",
        "https": "http://123.141.185.139:5031"
    },
    ProxyLocation.germany1: {
        "http": "http://161.35.70.249:8080",
        "https": "http://161.35.70.249:8080"
    },
    ProxyLocation.vietnam: {
        "http": "http://113.160.132.195:8080",
        "https": "http://113.160.132.195:8080"
    },
    ProxyLocation.hongkong: {
        "http": "http://91.103.120.39:80",
        "https": "http://91.103.120.39:80"
    },
    ProxyLocation.germany2: {
        "http": "http://57.129.81.201:8080",
        "https": "http://57.129.81.201:8080"
    }
}

# Pydantic model
class ProxyRequest(BaseModel):
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    body: Optional[str] = None

# Tạo app FastAPI
app = FastAPI(
    title="Proxy Request API",
    description="Forward HTTP requests through selected proxies",
    version="1.0.0"
)

# Bật CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proxy-api")

# Endpoint chính
@app.post("/proxy/")
async def proxy_request(
    target_url: HttpUrl = Query(..., description="Target URL to proxy to"),
    location: ProxyLocation = Query(ProxyLocation.korea),
    data: ProxyRequest = None
):
    proxies = PROXY_LIST[location]
    logger.info(f"Proxying {data.method} to {target_url} via {location} proxy")

    try:
        async with httpx.AsyncClient(proxies=proxies, timeout=30.0) as client:
            response = await client.request(
                method=data.method,
                url=str(target_url),
                headers=data.headers or {},
                content=data.body
            )

        return {
            "status_code": response.status_code,
            "content": response.text,
            "headers": dict(response.headers),
            "proxy_used": proxies,
            "your_apparent_ip": response.headers.get("X-Forwarded-For", "Unknown")
        }

    except httpx.RequestError as e:
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=502, detail="Failed to reach target URL through proxy")
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail=str(e))
