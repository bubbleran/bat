import os
#Envinroment variable to disable A2A SDK instrumentation
os.environ.setdefault("OTEL_INSTRUMENTATION_A2A_SDK_ENABLED", "false")
