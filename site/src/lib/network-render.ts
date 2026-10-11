import type { NetworkGraph } from "./network-graph";

export function escapeMarkup(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll("'", "&#39;");
}

/** The same SVG is used for the static page and the loaded model configuration. */
export function renderNetwork(graph: NetworkGraph, lang: "pt" | "en"): string {
  const e = escapeMarkup;
  const label = lang === "pt" ? "Arquitetura de geração do Transformer. Selecione uma operação para inspecionar." : "Transformer generation architecture. Select an operation to inspect it.";
  const kinds = ["flow", "residual", "cache", "shared", "feedback"];
  return `<svg class="network-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${graph.width} ${graph.height}" width="${graph.width}" height="${graph.height}" role="group" aria-label="${e(label)}">
    <defs>${kinds.map((kind) => `<marker id="network-arrow-${kind}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" class="arrow-${kind}"/></marker>`).join("")}</defs>
    ${graph.groups.map((g) => `<g class="network-group"><rect x="${g.x}" y="${g.y}" width="${g.width}" height="${g.height}" rx="14"/><text x="${g.x + 18}" y="${g.y + 25}" class="network-group-title">${e(g.title)}</text><text x="${g.x + 18}" y="${g.y + 44}" class="network-group-subtitle">${e(g.subtitle)}</text></g>`).join("")}
    ${graph.edges.map((edge) => `<g class="network-edge edge-${edge.kind}" data-network-edge="${e(edge.key)}" data-network-from="${e(edge.from)}" data-network-to="${e(edge.to)}"><path d="${e(edge.path)}" marker-end="url(#network-arrow-${edge.kind})"/>${edge.label ? `<text x="${edge.labelX}" y="${edge.labelY}" text-anchor="middle">${e(edge.label)}</text>` : ""}</g>`).join("")}
    ${graph.nodes.map((n) => `<g class="network-node node-${n.kind}" data-network-key="${e(n.key)}" transform="translate(${n.x} ${n.y})" role="button" tabindex="0" aria-label="${e(`${n.title}. ${n.subtitle}. ${n.shape}`)}" aria-pressed="false"><title>${e(`${n.title}\n${n.subtitle}\n${n.shape}`)}</title><rect width="${n.width}" height="${n.height}" rx="8"/><text x="${n.width / 2}" y="${n.height === 54 ? 16 : 22}" text-anchor="middle" class="network-node-title">${e(n.title)}</text><text x="${n.width / 2}" y="${n.height === 54 ? 32 : 40}" text-anchor="middle" class="network-node-equation">${e(n.subtitle)}</text><text x="${n.width / 2}" y="${n.height === 54 ? 45 : 56}" text-anchor="middle" class="network-node-shape">${e(n.shape)}</text></g>`).join("")}
  </svg>`;
}
