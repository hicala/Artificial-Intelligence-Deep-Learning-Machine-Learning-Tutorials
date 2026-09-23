"""The stub bodies P03-L07 hands the student, and the self-check answer key.

Author-side only: nothing here is copied into the student bundle. Each stub replaces the
solution body of the function it is keyed by, from the ATLAS_STUB_END marker to the end of
the function. The requirements, the worked example and the Returns line live in the comment
or docstring ABOVE the marker, so they are identical in the student file and the reference.
"""

SELF_CHECK_ANSWERS = {1: "c", 2: "b", 3: "d", 4: "c"}

C_STUBS = {
    "gw_alloc": """  // YOUR CODE HERE
  (void)bytes;
  todo("gw_alloc",
       "carve `bytes` out of g_arena at a cursor aligned up to 8, or return NULL and count "
       "the refusal; after gw_freeze() always return NULL");
  return NULL;""",

    "frame_decode": """  // YOUR CODE HERE
  (void)buf;
  (void)len;
  (void)out;
  todo("frame_decode",
       "check the length, then the function code, then the byte count, then the capacity, "
       "then the checksum, and only then fill *out with BIG-endian registers");
  return FD_SHORT;""",

    "q_health": """  // YOUR CODE HERE
  (void)x_q;
  (void)n;
  (void)baseline_q;
  todo("q_health",
       "sum the squares in int64_t, divide by n, take isqrt64 of that, then shift the result "
       "left by Q_FRAC in int64_t and divide by baseline_q");
  return -1;""",

    "sf_push": """  // YOUR CODE HERE
  (void)r;
  (void)machine;
  (void)hour;
  (void)wire;
  todo("sf_push",
       "refuse and count when the ring is full, changing nothing else; otherwise write at "
       "(head + count) % cap and return the sequence number you assigned");
  return SF_REFUSED;""",

    "sf_take": """  // YOUR CODE HERE
  (void)r;
  (void)out;
  (void)max;
  todo("sf_take",
       "copy up to max held records into out, oldest first, WITHOUT changing the ring");
  return 0;""",

    "sf_ack": """  // YOUR CODE HERE
  (void)r;
  (void)seq;
  todo("sf_ack",
       "drop every held record whose own seq is <= the one acknowledged, oldest first, and "
       "return how many were dropped");
  return 0;""",
}

PY_STUBS = {
    "float_health": """    # YOUR CODE HERE
    raise NotImplementedError("implement float_health")""",

    "agreement_report": """    # YOUR CODE HERE
    raise NotImplementedError("implement agreement_report")""",

    "psi": """    # YOUR CODE HERE
    raise NotImplementedError("implement psi")""",
}
