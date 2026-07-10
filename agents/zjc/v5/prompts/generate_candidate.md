---
name: generate_candidate
description: Conservative RGB underwater restoration candidate
version: 1.0
---

Restore this underwater image into a natural and physically plausible RGB image. Input 2 is only a conservative physics reference, not the target to copy. Produce a controlled but visible improvement beyond Input 2: cleaner haze, more natural local colors, and slightly richer missing chroma. Preserve all geometry, object boundaries, texture positions, and camera viewpoint. Do not add, remove, move, crop, rotate, stylize, sharpen, blur, or repaint objects. Do not over-brighten the foreground subject.
