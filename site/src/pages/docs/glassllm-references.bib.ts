import type { APIRoute } from "astro";
import { papers, paperBibtex } from "../../lib/docs-references";

export const GET: APIRoute = () => new Response(papers.map(paperBibtex).join("\n\n") + "\n", {
  headers: { "Content-Type": "application/x-bibtex; charset=utf-8", "Content-Disposition": 'attachment; filename="glassllm-references.bib"' },
});
