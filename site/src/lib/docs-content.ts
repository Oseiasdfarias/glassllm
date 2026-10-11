import type { MarkdownInstance } from "astro";
import type { Lang } from "../i18n";

export interface DocFrontmatter { title: string; description: string }
const documents = import.meta.glob<MarkdownInstance<DocFrontmatter>>("../content/docs/**/*.md", { eager: true });

export function getDocument(lang: Lang, slug: string): MarkdownInstance<DocFrontmatter> {
  const document = documents[`../content/docs/${lang}/${slug}.md`];
  if (!document?.frontmatter.title || !document.frontmatter.description || !document.rawContent().trim()) {
    throw new Error(`Missing complete documentation: ${lang}/${slug}`);
  }
  return document;
}
