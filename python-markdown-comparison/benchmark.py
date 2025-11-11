import timeit
import json
import markdown
import mistune
from markdown_it import MarkdownIt
import cmarkgfm

def run_benchmark(library_name, markdown_text):
    if library_name == "markdown":
        markdown.markdown(markdown_text)
    elif library_name == "mistune":
        mistune.html(markdown_text)
    elif library_name == "markdown-it-py":
        md = MarkdownIt()
        md.render(markdown_text)
    elif library_name == "cmarkgfm":
        cmarkgfm.github_flavored_markdown_to_html(markdown_text)

if __name__ == "__main__":
    with open("sample.md", "r") as f:
        markdown_text = f.read()

    libraries = ["markdown", "mistune", "markdown-it-py", "cmarkgfm"]
    results = {}

    for lib in libraries:
        stmt = f"run_benchmark('{lib}', markdown_text)"
        setup = f"from __main__ import run_benchmark, markdown_text"
        times = timeit.repeat(stmt, setup, repeat=10, number=10)
        results[lib] = min(times)

    with open("results.json", "w") as f:
        json.dump(results, f)
