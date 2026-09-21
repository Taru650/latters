"""The web application: a drafting page and an admin page.

Deliberately not a single-page app. FastAPI serves server-rendered HTML and
about sixty lines of plain JavaScript handle the two things that need it --
streaming a draft and updating a table in place.

No React, no build step, no CDN. The office machine has 3 GB of free RAM
that the language model needs, no Node toolchain, and **no internet**: a
`<script src="https://unpkg.com/...">` tag is a page that silently fails to
work there. Vendoring a framework would solve that and still cost more code
than writing the two interactions by hand.
"""
