# Multi-Provider LLM Support

DNA_RAG теперь поддерживает несколько LLM провайдеров с автоматическим переключением при сбоях (fallback).

## Поддерживаемые провайдеры

### OpenAI
- Модель по умолчанию: `gpt-4o-mini`
- API ключ: `OPENAI_API_KEY`
- Поддерживает все модели OpenAI API

### DeepSeek
- Модель по умолчанию: `deepseek-chat`
- API ключ: `DEEPSEEK_API_KEY`
- Поддерживает DeepSeek Chat и Code модели

## Конфигурация

### 1. Установка API ключей

Добавьте один или несколько API ключей в `.env` файл:

```bash
# OpenAI
OPENAI_API_KEY=sk-your-openai-key

# DeepSeek
DEEPSEEK_API_KEY=your-deepseek-key

# Или используйте префиксные версии
DNA_RAG_OPENAI_API_KEY=sk-your-openai-key
DNA_RAG_DEEPSEEK_API_KEY=your-deepseek-key
```

### 2. Установка приоритета провайдеров

Задайте список провайдеров в порядке приоритета (по умолчанию: `openai,deepseek`):

```bash
# OpenAI первый, DeepSeek как запасной
DNA_RAG_LLM_PROVIDERS=openai,deepseek

# DeepSeek первый, OpenAI как запасной
DNA_RAG_LLM_PROVIDERS=deepseek,openai

# Только OpenAI
DNA_RAG_LLM_PROVIDERS=openai
```

### 3. Выбор моделей

Настройте модели для каждого провайдера:

```bash
# OpenAI модель
DNA_RAG_OPENAI_MODEL=gpt-4o-mini  # или gpt-4, gpt-3.5-turbo

# DeepSeek модель
DNA_RAG_DEEPSEEK_MODEL=deepseek-chat  # или deepseek-coder
```

## Как это работает

### Приоритетное переключение (Priority Fallback)

Система автоматически выбирает провайдеров в указанном порядке приоритета:

1. **Первый провайдер**: Используется для всех запросов
2. **Второй провайдер**: Используется автоматически если первый недоступен
3. **И так далее**: Перебирает все настроенные провайдеры

### Пример сценария

Конфигурация:
```bash
OPENAI_API_KEY=sk-valid-key
DEEPSEEK_API_KEY=another-key
DNA_RAG_LLM_PROVIDERS=openai,deepseek
```

Поведение:
1. Система пытается использовать OpenAI (первый в приоритете)
2. Если OpenAI недоступен (сеть, лимиты, ошибка) → переключается на DeepSeek
3. Если DeepSeek тоже недоступен → выбрасывает ошибку

## Использование в коде

### Базовое использование

```python
from config import get_settings

# Загрузка конфигурации
settings = get_settings()

# Проверка что хотя бы один ключ настроен
settings.validate_api_keys()

# Создание менеджера провайдеров
llm_manager = settings.create_llm_manager()

# Генерация ответа
from llm_providers import LLMMessage

messages = [
    LLMMessage(role="user", content="Объясни что такое SNP")
]

response = llm_manager.generate(messages)
print(f"Provider: {response.provider}")
print(f"Response: {response.content}")
```

### Продвинутое использование

```python
from llm_providers import (
    OpenAIProvider,
    DeepSeekProvider,
    LLMProviderManager,
    LLMMessage
)

# Создание провайдеров вручную
providers = [
    OpenAIProvider(
        api_key="your-key",
        model="gpt-4",
        temperature=0.7,
        max_retries=3
    ),
    DeepSeekProvider(
        api_key="your-key",
        model="deepseek-chat",
        temperature=0.0,
        max_retries=2
    )
]

# Создание менеджера
manager = LLMProviderManager(providers)

# Генерация с дополнительными параметрами
messages = [LLMMessage(role="user", content="Test")]
response = manager.generate(messages, max_tokens=100)
```

## Логирование

Система логирует все переключения провайдеров:

```
INFO: Initialized LLM manager with 2 provider(s): ['openai', 'deepseek']
DEBUG: Attempting generation with openai
WARNING: Provider openai failed: Connection timeout
DEBUG: Attempting generation with deepseek
INFO: Successfully generated response using deepseek
```

## Обратная совместимость

Старый код продолжает работать:

```python
# Старый способ (все еще работает)
from config import get_settings
settings = get_settings()
api_key = settings.deepseek_api_key  # Получит DeepSeek или OpenAI ключ

# Новый способ (рекомендуется)
llm_manager = settings.create_llm_manager()
```

## Добавление новых провайдеров

Чтобы добавить новый провайдер:

1. Создайте класс, наследующий `BaseLLMProvider`:

```python
class MyCustomProvider(BaseLLMProvider):
    def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        # Ваша реализация
        pass

    def get_provider_name(self) -> str:
        return "mycustom"
```

2. Добавьте конфигурацию в `config.py`:

```python
mycustom_api_key: str = Field(default="", description="MyCustom API key")
mycustom_model: str = Field(default="model-name", description="MyCustom model")
```

3. Добавьте в `create_llm_manager`:

```python
elif provider_name == "mycustom" and self.mycustom_api_key:
    providers.append(MyCustomProvider(
        api_key=self.mycustom_api_key,
        model=self.mycustom_model,
        ...
    ))
```

## Тестирование

Запустите тесты для проверки конфигурации:

```bash
# Тесты провайдеров
pytest tests/test_llm_providers.py -v

# Тесты конфигурации
pytest tests/test_llm_config.py -v
```

## Часто задаваемые вопросы

### Можно ли использовать только один провайдер?

Да, просто укажите его в приоритете:
```bash
DNA_RAG_LLM_PROVIDERS=openai
OPENAI_API_KEY=your-key
```

### Что если оба провайдера недоступны?

Система выбросит ошибку `RuntimeError` с описанием последней ошибки.

### Как часто система переключается между провайдерами?

Каждый запрос начинается с первого провайдера в приоритете. Переключение происходит только при ошибке текущего провайдера.

### Можно ли задать разные параметры для разных провайдеров?

Да, используйте специфичные для провайдера настройки:
```bash
DNA_RAG_OPENAI_MODEL=gpt-4  # Для OpenAI
DNA_RAG_DEEPSEEK_MODEL=deepseek-coder  # Для DeepSeek
```

### Поддерживается ли кэширование?

Да, если провайдер вернул успешный ответ, он кэшируется на уровне приложения (не на уровне провайдера).
