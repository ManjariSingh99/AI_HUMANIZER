import os
from django.apps import AppConfig

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        if os.environ.get('RUN_MAIN') != 'true':
            return

        from ai_services.rewriter import TextRewriter
        print("🚀 Preloading T5 model...")
        TextRewriter()