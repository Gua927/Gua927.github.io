# Distill布局使用指南

## 概述

您的文章已经从 `post` 布局改为 `distill` 布局，这是一个专业的学术博客样式，类似于Yang Song的博客。

## Distill布局的特点

### 1. **专业的元数据展示**

```yaml
authors:
  - name: Runze Tian
    url: "https://gua927.github.io"
    affiliations:
      name: Renmin University of China
      url: "https://www.ruc.edu.cn"
```

在页面顶部会显示：

- **AUTHORS**: 作者姓名（可点击）
- **AFFILIATIONS**: 所属机构（可点击）
- **PUBLISHED**: 发布日期

### 2. **美观的标题和描述区域**

```yaml
title: The Unification of DDPM and Score-based Models
description: This post explores the unification of DDPM and Score-based Models...
```

标题会以大字体显示，下方是描述文字，提供文章概览。

### 3. **结构化的目录（TOC）**

```yaml
toc:
  - name: DDPM from a Score Perspective
  - name: SDE Model
    subsections:
      - name: Definition
      - name: Forward SDE
      - name: Reverse SDE
```

左侧会显示固定的目录导航，支持：

- 主要章节
- 子章节（subsections）
- 点击跳转
- 当前位置高亮

### 4. **学术风格的排版**

- 更宽的页面布局
- 优化的公式显示
- 脚注支持（d-footnote）
- 引用列表（d-citation）
- 附录区域（d-appendix）

## 与普通post布局的对比

| 特性     | Post布局       | Distill布局         |
| -------- | -------------- | ------------------- |
| 样式     | 简洁博客风格   | 学术论文风格        |
| 作者信息 | 简单显示       | 完整的作者/机构卡片 |
| 目录     | 侧边栏（可选） | 左侧固定，结构化    |
| 页面宽度 | 标准           | 更宽，适合技术内容  |
| 元数据   | 顶部简单信息   | 专业的分区展示      |
| 引用管理 | 基础支持       | 完整的学术引用系统  |

## 使用场景

### 适合Distill布局：

✅ 技术论文笔记  
✅ 研究总结  
✅ 深度技术教程  
✅ 算法解析  
✅ 学术性强的内容

### 适合Post布局：

✅ 日常博客  
✅ 短文  
✅ 项目介绍  
✅ 轻量级笔记

## 当前文章配置

您的文章现在使用了完整的Distill配置：

```yaml
---
layout: distill
title: The Unification of DDPM and Score-based Models
date: 2025-11-03 14:17:00
description: This post explores the unification of DDPM and Score-based Models in diffusion generative modeling...
authors:
  - name: Runze Tian
    url: "https://gua927.github.io"
    affiliations:
      name: Renmin University of China
      url: "https://www.ruc.edu.cn"
giscus_comments: true
related_posts: true
toc:
  - name: DDPM from a Score Perspective
  - name: SDE Model
    subsections:
      - name: Definition
      - name: Forward SDE
      - name: Reverse SDE
      - name: Optimization Target
      - name: PC Sampling
  - name: Probability Flow ODE
  - name: Conditional Generation
---
```

## 预览效果

构建站点后，您将看到：

### 页面顶部：

```
Generative Modeling by Estimating
Gradients of the Data Distribution

This post explores the unification of DDPM and Score-based Models
in diffusion generative modeling...

AUTHORS                AFFILIATIONS              PUBLISHED
Runze Tian            Renmin University         November 3, 2025
                      of China
```

### 左侧目录：

```
Contents
├─ DDPM from a Score Perspective
├─ SDE Model
│  ├─ Definition
│  ├─ Forward SDE
│  ├─ Reverse SDE
│  ├─ Optimization Target
│  └─ PC Sampling
├─ Probability Flow ODE
└─ Conditional Generation
```

## 高级特性

### 1. 添加脚注

在文章中使用：

```html
<d-footnote>这是一个脚注内容</d-footnote>
```

### 2. 添加引用

在文章中使用：

```html
<d-cite key="author2021paper"></d-cite>
```

### 3. 添加代码高亮

```html
<d-code block language="python"> def example(): return "Hello" </d-code>
```

### 4. 数学公式

完全兼容MathJax，支持：

- 行内公式：`$E = mc^2$`
- 显示公式：`$$...$$`

## 查看效果

运行本地服务器：

```bash
cd /root/Documents/Projects/Gua927.github.io
bundle exec jekyll serve
```

访问：`http://localhost:4000/blog/2025/The-Unification-of-DDPM-and-Score-based-Models/`

## 其他Distill文章示例

可以参考：

- Yang Song的博客：https://yang-song.net/blog/
- Distill.pub：https://distill.pub/
- 您可以查看这些站点的源码学习更多技巧

## 切换回普通布局

如果想切换回普通post布局，只需修改front matter：

```yaml
layout: post # 改回post
# 移除authors字段
# toc改为简单格式: toc: { sidebar: left }
```

---

**总结**：Distill布局为您的技术文章提供了专业、美观、学术化的展示方式，特别适合像您这样的深度技术内容！
