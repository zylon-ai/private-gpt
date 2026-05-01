"""Incremental ingestion module for PrivateGPT.

This module implements the proof-of-concept for incremental document updates
as described in the bachelor thesis. Instead of re-processing entire documents
when they change, this module:

1. Splits documents into semantic chunks (paragraph-based)
2. Computes SHA-256 hashes per chunk
3. Detects changes using diff algorithms (Myers / Patience)
4. Re-embeds only the changed chunks
5. Upserts changed chunks into the vector store

This avoids the "avalanche effect" of fixed-size chunking and dramatically
reduces the computational cost of updating modified documents.
"""
