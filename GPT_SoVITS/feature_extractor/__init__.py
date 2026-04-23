from . import cnhubert

content_module_map = {"cnhubert": cnhubert}

try:
    from . import whisper_enc
    content_module_map["whisper"] = whisper_enc
except ImportError:
    pass
