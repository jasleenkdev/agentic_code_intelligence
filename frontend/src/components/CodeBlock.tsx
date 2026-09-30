import clsx from 'clsx'
import { Highlight, themes } from 'prism-react-renderer'
import { useCurrentTheme } from '../lib/theme'

interface Props {
  code: string
  startLine: number
  highlight?: number[] | [number, number]
  highlightRange?: boolean // interpret `highlight` as an inclusive [from, to] range
  maxHeight?: string
  scrollToHighlight?: boolean
  language?: string // prism grammar: 'jsx' (Gladys, default) or 'python' (APPS)
}

/** Syntax-highlighted code (JSX by default) with real line numbers and highlighted lines. */
export function CodeBlock({ code, startLine, highlight = [], highlightRange, maxHeight,
  scrollToHighlight, language = 'jsx' }: Props) {
  const theme = useCurrentTheme()
  const isHl = (n: number) =>
    highlightRange && highlight.length === 2
      ? n >= highlight[0] && n <= highlight[1]
      : (highlight as number[]).includes(n)
  return (
    <Highlight
      theme={theme === 'dark' ? themes.vsDark : themes.github}
      code={code.replace(/\n$/, '')}
      language={language}
    >
      {({ tokens, getLineProps, getTokenProps }) => (
        <pre
          className="code-surface overflow-auto rounded-lg border border-app py-2 font-mono text-[12.5px] leading-[1.55]"
          style={{ maxHeight }}
        >
          {tokens.map((line, i) => {
            const n = startLine + i
            const hl = isHl(n)
            const { className, style, ...rest } = getLineProps({ line })
            return (
              <div
                key={i}
                {...rest}
                ref={hl && scrollToHighlight ? (el) => el?.scrollIntoView({ block: 'center' })
                  : undefined}
                className={clsx(className, 'flex px-3', hl && 'line-hl')}
                style={{ ...style, background: undefined }}
                data-line={n}
              >
                <span className="mr-4 inline-block w-10 shrink-0 select-none text-right text-muted opacity-70">
                  {n}
                </span>
                <span className="whitespace-pre">
                  {line.map((token, key) => {
                    const tp = getTokenProps({ token })
                    return <span key={key} {...tp} />
                  })}
                </span>
              </div>
            )
          })}
        </pre>
      )}
    </Highlight>
  )
}
