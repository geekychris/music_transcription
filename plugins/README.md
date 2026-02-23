# External Plugins

Place custom transcription plugin `.py` files in this directory.

Each plugin file should contain a class that inherits from `TranscriberPlugin`
and implements all abstract methods. Plugins are auto-discovered at startup.

## Example Plugin

```python
import numpy as np
from app.transcription.base import TranscriberPlugin, TranscriptionResult

class MyCustomPlugin(TranscriberPlugin):
    def name(self) -> str:
        return "My Custom Plugin"

    def description(self) -> str:
        return "A custom transcription plugin."

    def supports_realtime(self) -> bool:
        return False

    def transcribe(self, audio: np.ndarray, sr: int) -> TranscriptionResult:
        # Your transcription logic here
        return TranscriptionResult()
```
