# 网站数学公式和图像显示优化说明

## 优化内容

为了解决文章中公式过长导致页面过宽、图像大小问题以及公式显示错误的问题，已对网站进行以下优化：

## 1. 数学公式优化

### 文件修改：

- **`/assets/js/mathjax-setup.js`** - MathJax配置文件
- **`/_sass/_responsive-math.scss`** - 新建的响应式数学公式样式文件
- **`/assets/css/main.scss`** - 主样式文件（添加导入）

### 优化特性：

#### a) 长公式处理

- 为显示公式（`display="true"`）添加水平滚动功能
- 公式超出容器宽度时自动显示滚动条
- 自定义滚动条样式，更美观且不突兀

#### b) 行内公式优化

- 行内公式（`display="false"`）保持垂直对齐
- 允许非常长的行内公式自动换行

#### c) MathJax配置增强

```javascript
- processEscapes: true - 正确处理转义字符
- processEnvironments: true - 正确处理LaTeX环境
- matchFontHeight: true - 匹配字体高度
- scale: 1, minScale: 0.5 - 响应式缩放
```

## 2. 图像显示优化

### 优化特性：

#### a) 响应式图像

- 所有图像自动适应容器宽度（`max-width: 100%`）
- 高度自动调整保持原始比例（`height: auto`）
- 图像居中显示（`margin: 1.5rem auto`）

#### b) 视觉增强

- 添加圆角边框（`border-radius: 4px`）
- 添加轻微阴影效果，提升视觉层次
- GIF图像保持原始渲染质量

#### c) 移动端优化

- 小屏幕设备上减少图像边距
- 确保图像不会破坏页面布局

## 3. 暗色模式支持

- 为暗色模式单独定制滚动条样式
- 图像阴影在暗色模式下自动调整

## 4. 表格和代码块优化

- 表格支持水平滚动
- 代码块支持水平滚动
- 统一的滚动条样式

## 技术细节

### CSS选择器

```scss
// 显示公式
mjx-container[jax="CHTML"][display="true"]

// 行内公式
mjx-container[jax="CHTML"][display="false"]

// 通用MathJax容器
.MathJax, .MathJax_Display

// 文章图像
.post img

// 暗色模式
html[data-theme='dark']
```

### 滚动条样式

- WebKit浏览器（Chrome、Safari、Edge）：`::-webkit-scrollbar-*`
- Firefox浏览器：`scrollbar-width` 和 `scrollbar-color`
- 高度：6px（不占用太多空间）
- 颜色：使用主题颜色变量，自适应亮色/暗色模式

## 使用建议

### 1. 对于极长的公式

虽然现在支持水平滚动，但建议在markdown中使用以下方法提高可读性：

```latex
$$
\begin{aligned}
长公式 &= 第一部分 \\
&+ 第二部分 \\
&+ 第三部分
\end{aligned}
$$
```

### 2. 对于图像

- 建议使用适当分辨率的图像（不要过大）
- GIF动图会自动保持最佳渲染
- 外部图片链接会自动响应式处理

### 3. 测试建议

- 在不同屏幕尺寸下查看（使用浏览器开发者工具）
- 测试亮色/暗色模式切换
- 确认公式在移动设备上可以正常滚动

## 浏览器兼容性

- ✅ Chrome/Edge（完全支持）
- ✅ Firefox（完全支持）
- ✅ Safari（完全支持）
- ✅ 移动浏览器（完全支持）

## 文件结构

```
Gua927.github.io/
├── assets/
│   ├── css/
│   │   └── main.scss           # 主样式文件（已修改）
│   └── js/
│       └── mathjax-setup.js    # MathJax配置（已修改）
└── _sass/
    ├── _base.scss              # 基础样式（已清理）
    └── _responsive-math.scss   # 新增：响应式数学和内容优化
```

## 部署后验证

1. 清除浏览器缓存
2. 重新构建Jekyll站点：`bundle exec jekyll build`
3. 访问文章页面，检查：
   - 长公式是否可以水平滚动
   - 图像是否正确显示并居中
   - 页面宽度是否正常（不会过宽）
   - 滚动条样式是否美观

## 回滚方案

如果需要回滚更改：

1. 从`/_sass/_responsive-math.scss`删除或注释掉相关样式
2. 从`/assets/css/main.scss`中移除`"responsive-math"`导入
3. 恢复`/assets/js/mathjax-setup.js`到原始配置

---

**优化完成时间：** 2025-11-03  
**测试状态：** 待验证  
**兼容性：** 所有现代浏览器
