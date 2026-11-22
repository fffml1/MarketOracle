import { memo } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ElementContent, Root } from 'hast'

const evidencePrefix = '/__evidence/'
const evidencePattern = /\[(EV_[A-Z]+_\d+)\]/g

function linkEvidence() {
  function visit(parent: Root | Extract<ElementContent, { type: 'element' }>) {
    parent.children = parent.children.flatMap((child): ElementContent[] => {
      if (child.type === 'element' && !['a', 'code', 'pre'].includes(child.tagName)) visit(child)
      if (child.type !== 'text') return [child as ElementContent]
      const parts = child.value.split(evidencePattern)
      if (parts.length === 1) return [child]
      return parts.map((value, index) => index % 2 === 0 ? { type: 'text', value } : {
        type: 'element', tagName: 'a', properties: { href: `${evidencePrefix}${value}` },
        children: [{ type: 'text', value }],
      })
    })
  }
  return visit
}

export const MarkdownAnswer = memo(function MarkdownAnswer({ text, evidenceIds, onEvidence }: {
  text: string
  evidenceIds: Set<string>
  onEvidence: (id: string) => void
}) {
  return <div className="markdown-answer"><Markdown skipHtml disallowedElements={['img']}
    remarkPlugins={[remarkGfm]} rehypePlugins={[linkEvidence]} components={{
      a: ({ href, children }) => {
        if (href?.startsWith(evidencePrefix)) {
          const id = href.slice(evidencePrefix.length)
          return evidenceIds.has(id) ? <button type="button" className="citation" onClick={() => onEvidence(id)}>{id}</button> : <span>[{id}]</span>
        }
        return href ? <a href={href} target="_blank" rel="noopener noreferrer">{children}</a> : <span>{children}</span>
      },
      table: ({ children }) => <div className="markdown-table"><table>{children}</table></div>,
    }}>{text}</Markdown></div>
})
