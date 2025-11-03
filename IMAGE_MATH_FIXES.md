# 图像和数学公式修复说明

## 修复时间

2025-11-03

## 问题分析

### 1. 行内公式编译错误

**问题原因：**

- 文章中使用了 `\(c = 1\)` 作为行内公式标记
- Jekyll/Kramdown的MathJax配置默认支持 `$...$` 和 `\(...\)`
- 但在某些上下文中，`\(` 可能不被正确识别或与Markdown语法冲突

**具体位置：**
在 "Brownian Motion" 定义部分：

```markdown
If \(c = 1\), it is called **standard Brownian motion**
```

**解决方案：**
统一使用 `$...$` 作为行内公式标记：

```markdown
If $c = 1$, it is called **standard Brownian motion**
```

**为什么这样修复有效：**

1. `$...$` 是MathJax配置中明确定义的行内数学分隔符
2. 在Jekyll/Kramdown中更稳定可靠
3. 与文章中其他所有行内公式保持一致

### 2. 图像大小控制问题

**问题原因：**

- 使用标准Markdown图片语法 `![alt](url)` 无法精确控制图片大小
- CSS样式中 `.post img` 的 `max-width: 100%` 只能限制最大宽度
- 外部图片（如CSDN托管的截图）原始尺寸过大，导致显示过大

**解决方案：**
将所有Markdown图片格式改为Liquid模板格式：

**修改前（Markdown格式）：**

```markdown
![img](https://yang-song.net/assets/img/score/denoise_vp.gif)
```

**修改后（Liquid格式）：**

```markdown
{% include figure.liquid loading="eager" path="https://yang-song.net/assets/img/score/denoise_vp.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
```

### Liquid图片标签的优势

#### 1. Bootstrap响应式类支持

- `img-fluid`：自动响应式调整，确保在不同屏幕尺寸下都能正确显示
- `rounded`：添加圆角效果
- `z-depth-1`：添加Material Design风格的阴影

#### 2. 更好的加载控制

- `loading="eager"`：对于文章顶部的图片优先加载
- 可选 `loading="lazy"`：延迟加载，提高页面性能

#### 3. 图片缩放功能

- `zoomable=true`：点击图片可以放大查看
- 提供更好的用户体验，特别是对于算法流程图等细节丰富的图片

#### 4. 一致的样式

- 所有图片样式统一，由theme的 `_includes/figure.liquid` 统一控制
- 便于全局修改和维护

## 修改的图片列表

共修改了6张图片：

1. **denoise_vp.gif** - Diffusion过程动画

   - 原始：`![img](https://yang-song.net/assets/img/score/denoise_vp.gif)`
   - 位置：SDE Model章节

2. **sde_schematic.jpg** - SDE示意图

   - 原始：`![img](https://yang-song.net/assets/img/score/sde_schematic.jpg)`
   - 位置：Reverse SDE章节

3. **DDPM算法图** - Ancestral Sampling

   - 原始：`![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/11a3ad2f3090305f3e0f40a64e197955.png#pic_center)`
   - 位置：PC Sampling章节

4. **NCSN算法图** - Langevin Dynamics

   - 原始：`![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e761207ab3238ab653ee29b6b780ce67.png#pic_center)`
   - 位置：PC Sampling章节

5. **PC Sampling算法图** - Predictor-Corrector

   - 原始：`![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/57e5ddf18d1b3573d7aa65539109b823.png)`
   - 位置：PC Sampling章节

6. **teaser.jpg** - ODE轨迹对比图
   - 原始：`![img](https://yang-song.net/assets/img/score/teaser.jpg)`
   - 位置：Probability Flow ODE章节

## 验证方法

### 1. 本地测试

```bash
cd /root/Documents/Projects/Gua927.github.io
bundle exec jekyll serve
```

访问 `http://localhost:4000` 查看文章

### 2. 检查要点

- ✅ 所有行内公式正确渲染（特别是 "c = 1" 部分）
- ✅ 图片大小适中，不会过大
- ✅ 图片居中显示
- ✅ 图片可以点击放大
- ✅ 响应式：在不同屏幕尺寸下测试

### 3. 浏览器测试

- Desktop: 宽屏显示效果
- Tablet: 中等屏幕适配
- Mobile: 移动端显示

## 技术细节

### MathJax配置（已在 assets/js/mathjax-setup.js 中）

```javascript
tex: {
  tags: "ams",
  inlineMath: [
    ["$", "$"],        // ✅ 推荐使用
    ["\\(", "\\)"],    // ⚠️  可能在某些情况下失败
  ],
}
```

### 图片样式（已在 \_sass/\_responsive-math.scss 中）

```scss
.post img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 1.5rem auto;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
```

### Figure Liquid模板（\_includes/figure.liquid）

该模板自动处理：

- 响应式布局
- 图片加载优化
- 缩放功能
- 统一样式

## 最佳实践

### 1. 行内公式

✅ **推荐：** 使用 `$...$`

```markdown
The value is $c = 1$ in this case.
```

❌ **避免：** 使用 `\(...\)`（虽然有时可以工作）

```markdown
The value is \(c = 1\) in this case.
```

### 2. 显示公式

✅ **推荐：** 使用 `$$...$$`

```markdown
$$
E = mc^2
$$
```

### 3. 图片插入

✅ **推荐：** 使用Liquid模板

```liquid
{%
  include figure.liquid
  loading="eager"
  path="图片URL"
  class="img-fluid rounded z-depth-1"
  zoomable=true
  caption="可选的图片说明"
%}
```

❌ **避免：** 使用纯Markdown（除非是简单场景）

```markdown
![alt](url)
```

## 相关文件

- 文章文件：`_posts/2025-11-03-Note-Diffusion-DDPM&NCSN.md`
- 样式文件：`_sass/_responsive-math.scss`
- MathJax配置：`assets/js/mathjax-setup.js`
- 图片模板：`_includes/figure.liquid`

## 总结

通过这次修复，我们：

1. ✅ 统一了行内公式的标记方式，避免渲染错误
2. ✅ 将所有图片改为Liquid格式，实现更好的大小控制
3. ✅ 增强了图片的交互性（缩放功能）
4. ✅ 提高了页面的响应式表现
5. ✅ 保持了代码的一致性和可维护性

---

**修复者：** GitHub Copilot  
**日期：** 2025-11-03  
**状态：** ✅ 已完成
