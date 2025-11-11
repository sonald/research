import json
import matplotlib.pyplot as plt

def plot_results(results):
    libraries = list(results.keys())
    times = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(libraries, times, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Markdown Library')
    plt.ylabel('Time (seconds)')
    plt.title('Python Markdown Library Performance')
    plt.savefig('benchmark.png')

if __name__ == "__main__":
    with open("results.json", "r") as f:
        results = json.load(f)
    plot_results(results)
