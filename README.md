# DAR project page

这是一个无需后端、无需安装任何软件的静态论文主页。双击 `index.html` 可以在浏览器中预览；发布时将整个文件夹上传到 GitHub Pages 即可。

## 当前已写入的内容

- 论文标题、作者、单位、摘要
- DAR 数据规模、任务设置、训练/测试划分
- DAR-R1 的两阶段训练流程
- 主结果表与人类评价结果
- 论文引用 BibTeX
- 适配手机、平板、电脑的响应式布局
- 参照你给出的 SceneScribe-1M 页面采用“标题—摘要—图示—数据集—方法—结果—样例—引用”的学术项目页结构，但没有直接复制其源码或素材。

## 目前需要你后续补的文件

论文给我的附件只有 `.tex`，没有论文 PDF、图片或视频。因此页面中下列位置目前是完整排版的可替换示意：

1. `assets/paper/DAR.pdf`：放入论文最终 PDF 后，把首页的 `Paper` 按钮链接从 `#citation` 改成 `assets/paper/DAR.pdf`。
2. `assets/media/`：放 3 个 MP4 或 WebM demo 视频，并替换 `index.html` 中 `id="examples"` 的 3 个占位卡片。
3. Code 和 Dataset：公开后把两个 `coming soon` 按钮改成对应仓库 / 数据集链接。
4. 若你有论文中的 `task.pdf`、`dataset_pipeline.pdf`、`MLLM_outputs.pdf` 等正式图片，可以放到 `assets/images/`，再将页面 SVG 图替换为论文图。现有 SVG 图是为了让你没有原始图片时也能立即获得一个可用页面。

## 最简单的 GitHub Pages 发布方法

1. 登录 GitHub，进入 `https://github.com/Zhang-Zhiyan`。
2. 点击右上角 **New repository**，仓库名建议填 `DAR`，选择 **Public**，然后创建。
3. 解压本文件夹，打开解压后的 `DAR-project-page` 文件夹，选中其中的所有内容（包括 `index.html`、`assets`、`.nojekyll`、`README.md`），上传到新仓库的根目录。注意不要把整个 `DAR-project-page/` 文件夹作为一层再上传。
4. 打开仓库的 **Settings → Pages**。
5. 在 **Build and deployment** 中选择：
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/(root)`
6. 点击 Save。GitHub 通常会在几分钟内给出你的网址：

```text
https://zhang-zhiyan.github.io/DAR/
```

如果将来要把这个项目放到个人主页根目录，可创建名为 `Zhang-Zhiyan.github.io` 的仓库；但现在单独建 `DAR` 仓库更清楚，也不会影响现有 GitHub 页面。

## 你最常改的 4 个位置

用 VS Code 打开 `index.html` 后，搜索：

- `ECCV 2026 · Project Page`：会议 / 状态标签
- `Code · coming soon`：代码链接
- `Dataset · coming soon`：数据链接
- `@inproceedings{zhang2026dar`：BibTeX

## 本地预览

最简单：直接双击 `index.html`。

如果浏览器对本地文件有任何限制，在文件夹内打开终端运行：

```bash
python -m http.server 8000
```

再访问：

```text
http://localhost:8000
```
