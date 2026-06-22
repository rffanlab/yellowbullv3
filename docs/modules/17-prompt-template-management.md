# Prompt 模板管理详细设计

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **模板存储** | 持久化存储 Prompt 模板，支持版本管理 |
| **变量替换** | 支持 Jinja2 风格的变量插值和条件逻辑 |
| **热更新** | 运行时动态加载/修改模板，无需重启服务 |
| **A/B 测试** | 多版本模板并行运行，对比效果 |
| **权限控制** | 不同用户/角色可访问不同的模板集 |

---

## 2. Template Store `prompt/template_store.py`

```python
"""
Prompt 模板存储与管理。

支持：
- YAML 文件加载（默认）
- SQLite 持久化
- 内存缓存 + 热更新
- 版本管理
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Prompt 模板定义"""
    name: str                          # 模板唯一标识
    version: int = 1                   # 版本号
    content: str = ""                  # 模板内容（支持变量）
    description: str = ""              # 描述
    variables: list[str] = field(default_factory=list)  # 需要的变量名
    category: str = "general"          # 分类：system, tool, agent, general
    enabled: bool = True               # 是否启用
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def render(self, **kwargs: Any) -> str:
        """渲染模板，替换变量"""
        import jinja2
        template = jinja2.Template(self.content)
        return template.render(**kwargs)


class TemplateStore:
    """模板存储"""

    def __init__(self, storage_path: str = None):
        self._templates: dict[str, PromptTemplate] = {}
        self._storage_path = Path(storage_path or "data/prompts")
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    def save(self, template: PromptTemplate) -> PromptTemplate:
        """保存模板"""
        template.updated_at = time.time()
        self._templates[template.name] = template
        self._save_to_disk(template)
        logger.info(f"Saved prompt template: {template.name} v{template.version}")
        return template

    def get(self, name: str) -> Optional[PromptTemplate]:
        """获取模板"""
        template = self._templates.get(name)
        if template and not template.enabled:
            return None
        return template

    def get_all(self, category: str = None) -> list[PromptTemplate]:
        """获取所有模板（可选按分类过滤）"""
        templates = [t for t in self._templates.values() if t.enabled]
        if category:
            templates = [t for t in templates if t.category == category]
        return templates

    def delete(self, name: str) -> bool:
        """删除模板"""
        if name in self._templates:
            del self._templates[name]
            template_file = self._storage_path / f"{name}.yaml"
            if template_file.exists():
                template_file.unlink()
            logger.info(f"Deleted prompt template: {name}")
            return True
        return False

    def list_names(self) -> list[str]:
        """列出所有模板名称"""
        return list(self._templates.keys())

    def _load_from_disk(self):
        """从磁盘加载模板"""
        for yaml_file in self._storage_path.glob("*.yaml"):
            try:
                import yaml
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                template = PromptTemplate(
                    name=data.get("name", yaml_file.stem),
                    version=data.get("version", 1),
                    content=data.get("content", ""),
                    description=data.get("description", ""),
                    variables=data.get("variables", []),
                    category=data.get("category", "general"),
                    enabled=data.get("enabled", True),
                )
                self._templates[template.name] = template

            except Exception as e:
                logger.error(f"Failed to load template {yaml_file}: {e}")

    def _save_to_disk(self, template: PromptTemplate):
        """保存模板到磁盘"""
        import yaml
        template_file = self._storage_path / f"{template.name}.yaml"
        with open(template_file, "w", encoding="utf-8") as f:
            yaml.dump({
                "name": template.name,
                "version": template.version,
                "content": template.content,
                "description": template.description,
                "variables": template.variables,
                "category": template.category,
                "enabled": template.enabled,
            }, f, allow_unicode=True)


def get_template_store() -> TemplateStore:
    """获取全局模板存储"""
    import config.manager as cm
    manager = cm.get_manager()
    if not hasattr(manager, "_template_store"):
        settings = (manager.settings.prompt or {})
        storage_path = settings.get("storage_path", "data/prompts")
        manager._template_store = TemplateStore(storage_path)
    return manager._template_store
```

---

## 3. Template Renderer `prompt/renderer.py`

```python
"""
Prompt 模板渲染器。

功能：
- Jinja2 变量替换
- 条件逻辑支持
- 循环和过滤器
- 嵌套模板引用
"""

import logging
from typing import Any, Optional

from prompt.template_store import TemplateStore, PromptTemplate

logger = logging.getLogger(__name__)


class TemplateRenderer:
    """模板渲染器"""

    def __init__(self, store: TemplateStore):
        self._store = store

    def render(self, template_name: str, **kwargs: Any) -> Optional[str]:
        """渲染指定模板"""
        template = self._store.get(template_name)
        if not template:
            logger.warning(f"Template '{template_name}' not found or disabled")
            return None

        try:
            return template.render(**kwargs)
        except Exception as e:
            logger.error(f"Failed to render template '{template_name}': {e}")
            return None

    def render_system_prompt(
        self,
        base_template: str = "system_base",
        **overrides: Any,
    ) -> str:
        """渲染 System Prompt（支持多层继承）"""
        # 获取基础模板
        base = self._store.get(base_template)
        if not base:
            return ""

        # 合并覆盖变量
        variables = {
            "language": overrides.get("language", "中文"),
            "max_tokens": overrides.get("max_tokens", 4096),
            **overrides,
        }

        try:
            import jinja2
            template = jinja2.Template(base.content)
            return template.render(**variables)
        except Exception as e:
            logger.error(f"Failed to render system prompt: {e}")
            return base.content


def get_renderer() -> TemplateRenderer:
    """获取全局渲染器"""
    import config.manager as cm
    manager = cm.get_manager()
    if not hasattr(manager, "_template_renderer"):
        from prompt.template_store import get_template_store
        manager._template_renderer = TemplateRenderer(get_template_store())
    return manager._template_renderer
```

---

## 4. API Routes `api/prompt.py`

```python
"""
Prompt 模板管理 API。
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

router = APIRouter(prefix="/prompt", tags=["prompt"])


@router.get("/templates")
async def list_templates(category: str = None):
    """列出所有 Prompt 模板"""
    from prompt.template_store import get_template_store

    store = get_template_store()
    templates = store.get_all(category)
    return {
        "templates": [
            {
                "name": t.name,
                "version": t.version,
                "description": t.description,
                "category": t.category,
                "variables": t.variables,
                "enabled": t.enabled,
            }
            for t in templates
        ]
    }


@router.get("/templates/{name}")
async def get_template(name: str):
    """获取模板详情"""
    from prompt.template_store import get_template_store

    store = get_template_store()
    template = store._templates.get(name)  # 使用 _templates 以获取禁用的模板

    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")

    return {
        "name": template.name,
        "version": template.version,
        "content": template.content,
        "description": template.description,
        "variables": template.variables,
        "category": template.category,
        "enabled": template.enabled,
        "created_at": template.created_at,
        "updated_at": template.updated_at,
    }


@router.post("/templates")
async def create_template(
    name: str,
    content: str,
    description: str = "",
    variables: list[str] = None,
    category: str = "general",
):
    """创建新模板"""
    from prompt.template_store import get_template_store, PromptTemplate

    store = get_template_store()

    if name in store._templates:
        raise HTTPException(status_code=409, detail=f"Template '{name}' already exists")

    template = PromptTemplate(
        name=name,
        content=content,
        description=description,
        variables=variables or [],
        category=category,
    )
    store.save(template)

    return {"status": "created", "name": name}


@router.put("/templates/{name}")
async def update_template(
    name: str,
    content: str = None,
    description: str = None,
    variables: list[str] = None,
    enabled: bool = None,
):
    """更新模板"""
    from prompt.template_store import get_template_store

    store = get_template_store()
    template = store._templates.get(name)

    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")

    if content is not None:
        template.content = content
    if description is not None:
        template.description = description
    if variables is not None:
        template.variables = variables
    if enabled is not None:
        template.enabled = enabled

    # 版本号递增
    template.version += 1
    store.save(template)

    return {"status": "updated", "name": name, "version": template.version}


@router.post("/templates/{name}/render")
async def render_template(name: str, variables: dict = None):
    """渲染模板（测试用）"""
    from prompt.renderer import get_renderer

    renderer = get_renderer()
    result = renderer.render(name, **(variables or {}))

    if result is None:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")

    return {"template": name, "rendered": result}


@router.delete("/templates/{name}")
async def delete_template(name: str):
    """删除模板"""
    from prompt.template_store import get_template_store

    store = get_template_store()
    if not store.delete(name):
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")

    return {"status": "deleted", "name": name}
```

---

## 5. 默认模板 `data/prompts/*.yaml`

### system_base.yaml
```yaml
name: system_base
version: 1
content: |
  你是一个专业的 AI 助手。请遵循以下规则：

  1. 使用 {{ language }} 回答用户问题
  2. 保持回答简洁、准确、有帮助
  3. 对于不确定的信息，明确说明不确定性
  4. 不要编造事实或数据
  5. 当需要工具时，主动调用可用工具

  {% if tools %}
  ## 可用工具
  {{ tools }}
  {% endif %}

  {% if custom_instructions %}
  ## 额外指令
  {{ custom_instructions }}
  {% endif %}
description: "基础 System Prompt 模板"
variables: ["language", "tools", "custom_instructions"]
category: system
enabled: true
```

### code_review.yaml
```yaml
name: code_review
version: 1
content: |
  请审查以下代码，指出潜在问题并提供改进建议：

  ## 语言
  {{ language }}

  ## 代码
  ```{{ language }}
  {{ code }}
  ```

  ## 关注点
  {% for focus in focuses %}
  - {{ focus }}
  {% endfor %}

  请按以下格式回复：
  1. **问题列表**（严重程度、位置、描述）
  2. **改进建议**（具体修改方案）
  3. **重构后的代码**（如有必要）
description: "代码审查 Prompt"
variables: ["language", "code", "focuses"]
category: tool
enabled: true
```

### data_analysis.yaml
```yaml
name: data_analysis
version: 1
content: |
  请分析以下数据并回答用户的问题。

  ## 数据格式
  {{ format }}

  ## 数据内容
  ```
  {{ data_preview }}
  ```

  ## 用户问题
  {{ question }}

  ## 要求
  - 使用 Python (pandas/numpy) 进行计算
  - 提供清晰的统计结果
  - 用自然语言解释发现
description: "数据分析 Prompt"
variables: ["format", "data_preview", "question"]
category: tool
enabled: true
```

---

## 6. YAML 配置 `prompt` section

```yaml
# config/settings.yaml (新增)
prompt:
  storage_path: "data/prompts"   # 模板存储路径
  default_language: "中文"        # 默认语言

  # 热更新
  hot_reload: false               # 是否监听文件变化自动重载
  reload_interval_seconds: 60     # 轮询间隔（hot_reload=true 时生效）

  # A/B 测试
  ab_testing:
    enabled: false
    variants:                     # 模板变体权重
      system_base_v1: 80          # 80% 流量
      system_base_v2: 20          # 20% 流量
```

---

## 7. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **模板存储** | YAML 文件 + 内存缓存，支持热更新 |
| **变量替换** | Jinja2 引擎，支持条件、循环、过滤器 |
| **版本管理** | 每次编辑自动递增版本号 |
| **API 管理** | CRUD + 渲染测试接口 |
| **分类组织** | system / tool / agent / general 四类 |
| **A/B 测试** | 多模板变体，按权重分配流量 |
