# Speculative Decoding Lab

一个单文件、无依赖的交互式研究演示，比较 DFlash、DDTree 和 DSpark。

直接打开 `index.html`，或在本目录运行：

```bash
python3 -m http.server 8000
```

再访问 <http://localhost:8000>。

页面里的概率、接受长度和交互参数用于解释机制，不是 benchmark。性能结论应回到页面链接的一手论文，并结合模型、GPU、batch/concurrency 与 serving profile 解读。
