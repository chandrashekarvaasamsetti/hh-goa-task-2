import time
import statistics

from retrieval.search import search


QUERIES = [
    "కంపెనీ ఏ రాష్ట్రంలో చేర్చబడుతుంది?",
    "కార్పొరేషన్ అంటే ఏమిటి?",
    "కంపెనీ ఎలా నిర్వహించబడుతుంది?",
    "ప్రభుత్వ యాజమాన్యంలోని సంస్థ అంటే ఏమిటి?",
    "కంపెనీ స్టాక్‌ను జారీ చేయవచ్చా?",
    "కంపెనీ వాటాదారులచే ఎలా పాలించబడుతుంది?",
    "కార్పొరేషన్ చట్టపరంగా ఎలా వ్యవహరిస్తుంది?",
    "ఒక కంపెనీ ఎక్కడ చేర్చబడుతుంది?",
    "కంపెనీకి సంబంధించిన సమాచారం ఏమిటి?",
    "వాటాదారులు కంపెనీని ఎలా పాలిస్తారు?",
]


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0.0

    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    weight = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * weight
    )


def main():

    print("Warming up retrieval model...")

    search(QUERIES[0], top_k=5)

    latencies = []

    print()
    print("Running latency benchmark...")
    print()

    for i, query in enumerate(QUERIES, start=1):

        start = time.perf_counter()

        results = search(
            query,
            top_k=5
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed_ms)

        print(
            f"{i:02d}. "
            f"{elapsed_ms:.2f} ms | "
            f"results={len(results)}"
        )

    p50 = percentile(latencies, 0.50)
    p70 = percentile(latencies, 0.70)
    p100 = max(latencies)

    print()
    print("================================")
    print("RETRIEVAL LATENCY RESULTS")
    print("================================")
    print(f"Queries: {len(latencies)}")
    print(f"Minimum: {min(latencies):.2f} ms")
    print(f"Average: {statistics.mean(latencies):.2f} ms")
    print(f"P50: {p50:.2f} ms")
    print(f"P70: {p70:.2f} ms")
    print(f"P100: {p100:.2f} ms")
    print("================================")


if __name__ == "__main__":
    main()