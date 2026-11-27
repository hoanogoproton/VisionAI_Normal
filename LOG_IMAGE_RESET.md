# Image boundaries with independent acquisition

A valid product change or repeat log scan increments the main job epoch and each
active watcher's generation. Queued/leased old packets are explicitly invalidated
and counted; decoded cached images are cleared. Files already present at reset
are treated as old. New files or versions observed later can be collected.

A source read already running cannot be safely cancelled. Its old-generation
snapshot is discarded when the read completes. An inference already running
finishes on the single AI worker; its obsolete result does not reach GUI/COM.

This preserves the original "ignore existing images on accepted scan" policy,
not an exactly-once product association protocol. A valid image that is already
present before reset is intentionally ignored. Polling cannot see a version that
appears and disappears entirely between scans. Exact association/loss prevention
requires producer frame/batch IDs and a retention/acknowledgement protocol.

No source images or temporary copies are deleted. Shutdown and product-boundary
discards are logged explicitly. See ASYNC_READING_VI.md for deployment and limits.
