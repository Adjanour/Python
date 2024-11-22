import asyncio
import httpx
import time
from statistics import mean
from tabulate import tabulate
import json

# Configuration
API_URL = "https://askmeapi-1.onrender.com/api/ask"  # Replace with your API endpoint
CONCURRENT_REQUESTS = 2
data = '''
{
    "question": "How can I use oziza?",
    "conversation_history": [
        {"role": "user", "content": "Hi.", "createdAt": "2024-11-12T10:00:00Z"}
    ]
}
'''
# Parse the JSON string into a Python dictionary
TEXT_QUERY = json.loads(data)
API_KEY = "4d13f64dcd5c385239bbffedd1cf5103"  # Replace with your actual API key
OUTPUT_FILE = "benchmark_results.json"  # Output file for detailed results


async def make_request(client, index):
    """
    Makes a single request to the API and captures performance metrics.

    Args:
        client: HTTPX async client.
        index: Index of the request for logging purposes.

    Returns:
        A dictionary with latency, status code, and other metrics.
    """
    try:
        headers = {"x-api-key": API_KEY}  # Add the API key header
        start_time = time.perf_counter()
        async with client.stream("POST", API_URL, json={"question":"Hello what is oziza?"}, headers=headers, timeout=1000) as response:
            elapsed_time = time.perf_counter() - start_time

            if response.status_code == 200:
                # Stream and process content as it arrives
                content_chunks = []
                async for chunk in response.aiter_text():
                    content_chunks.append(chunk)

                full_content = "".join(content_chunks)
                content_size = len(full_content)
            else:
                full_content = None
                content_size = 0

        return {
            "request_id": index,
            "latency": elapsed_time,
            "status": response.status_code,
            "content_size": content_size,
            "streamed_content": full_content[:100],  # Store a sample of the streamed content
        }
    except Exception as e:
        return {
            "request_id": index,
            "latency": None,
            "status": "Failed",
            "content_size": 0,
            "error": str(e),
        }


async def benchmark():
    """
    Runs the benchmark by simulating multiple concurrent requests.
    """
    print(f"Starting benchmark: {CONCURRENT_REQUESTS} concurrent requests to {API_URL}")

    async with httpx.AsyncClient() as client:
        tasks = [make_request(client, i) for i in range(CONCURRENT_REQUESTS)]
        results = await asyncio.gather(*tasks)

    # Analyze results
    latencies = [r["latency"] for r in results if r["latency"] is not None]
    success_count = sum(1 for r in results if r["status"] == 200)
    failed_count = sum(1 for r in results if r["status"] != 200)

    # Tabulate results
    summary = [
        ["Total Requests", CONCURRENT_REQUESTS],
        ["Successful Requests", success_count],
        ["Failed Requests", failed_count],
        ["Average Latency (ms)", mean(latencies) * 1000 if latencies else "N/A"],
        ["Max Latency (ms)", max(latencies) * 1000 if latencies else "N/A"],
        ["Min Latency (ms)", min(latencies) * 1000 if latencies else "N/A"],
        ["Throughput (req/s)", len(latencies) / sum(latencies) if latencies else "N/A"],
    ]

    # Print Summary
    print("\nBenchmark Summary:")
    print(tabulate(summary, headers=["Metric", "Value"], tablefmt="grid"))

    # Write Detailed Results to a File
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed results have been saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(benchmark())
