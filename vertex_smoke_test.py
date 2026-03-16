#!/usr/bin/env python3
"""
Vertex AI Gemini Pro Smoke Test (No API Key)
Based on requirements from vertex_gemini_pro_smoke_test_v2.pdf

This script validates that the VM/service account can access Vertex AI
using Application Default Credentials (ADC).

Pre-requisites:
1) Vertex AI API enabled in the target project
2) VM must have an attached service account with roles/aiplatform.user
3) Python 3.10+ available on the VM

Usage:
    export PROJECT_ID="jlr-dl-iqm"
    export LOCATION="europe-west2"
    export MODEL="gemini-2.0-flash-exp"
    python vertex_smoke_test.py "Reply with: Vertex smoke test OK"
"""

import os
import sys
import google.auth
from google import genai


def must_env(name: str) -> str:
    """Get required environment variable or exit"""
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing env var: {name}")
    return v


def main() -> None:
    # Required
    project_id = must_env("PROJECT_ID")

    # Optional (Vertex regions: e.g., us-central1, europe-west4, etc.)
    location = os.getenv("LOCATION", "europe-west2")

    # Application Default Credentials (ADC)
    # On a GCP VM, this comes from the VM's attached service account (no API key needed).
    creds, adc_project = google.auth.default()
    print(f"[auth] Using ADC. adc_project={adc_project!r} target_project={project_id!r}")

    # Vertex AI backend (not Gemini Developer API)
    client = genai.Client(vertexai=True, project=project_id, location=location)

    # Pro model (Vertex AI)
    model = os.getenv("MODEL", "gemini-2.0-flash-exp")

    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        prompt = "Reply with: 'Vertex smoke test OK' and the region you're running in."

    print(f"[request] model={model} location={location}")

    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        print("\n=== MODEL RESPONSE ===")
        print(resp.text)
        print("\n✅ Smoke test PASSED")

    except Exception as e:
        print(f"\n❌ Smoke test FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
