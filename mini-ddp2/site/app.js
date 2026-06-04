const links = [...document.querySelectorAll(".toc a")];
const sections = links.map((link) => document.querySelector(link.getAttribute("href")));
const progress = document.querySelector(".progress span");

const observer = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    links.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
    });
  },
  { rootMargin: "-20% 0px -60% 0px", threshold: [0.1, 0.3, 0.6] },
);

sections.filter(Boolean).forEach((section) => observer.observe(section));

document.addEventListener("scroll", () => {
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  const ratio = scrollable <= 0 ? 0 : window.scrollY / scrollable;
  progress.style.width = `${Math.min(100, Math.max(0, ratio * 100))}%`;
});

document.querySelectorAll(".copy-btn").forEach((button) => {
  button.addEventListener("click", async () => {
    const block = button.closest(".code-block");
    const code = block?.querySelector("code")?.innerText ?? "";
    await navigator.clipboard.writeText(code);
    button.textContent = "Copied";
    setTimeout(() => {
      button.textContent = "Copy";
    }, 1200);
  });
});
