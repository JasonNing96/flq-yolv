在 Linux/WSL 环境下 Matplotlib 中文显示的最佳实践 (基于 plot.py 验证):
1. **策略**：不再依赖下载字体或全局配置，而是优先自动检测系统内置的高质量中文字体。
2. **检测顺序**：
   - `/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc` (首选，显示效果最好)
   - `/usr/share/fonts/truetype/arphic/uming.ttc` (常见备选)
   - `/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf` (兜底系统字体)
   - `./SimHei.ttf` (最后尝试当前目录下的文件)
3. **核心代码模式**：
   ```python
   from matplotlib.font_manager import FontProperties
   from pathlib import Path

   # 自动寻找可用字体路径
   font_path = '...' # 根据上述顺序检测到的第一个存在的路径
   
   # 创建字体属性对象
   cn_font = FontProperties(fname=font_path, size=12)
   
   # 显式应用 (比 rcParams 更稳定)
   plt.title("中文标题", fontproperties=cn_font)
   plt.xlabel("X轴", fontproperties=cn_font)
   plt.legend(prop=cn_font) # 注意 legend 使用 prop 参数
   ```
4. **优势**：兼容性强，无需手动安装字体即可在大多数环境中正常工作。
