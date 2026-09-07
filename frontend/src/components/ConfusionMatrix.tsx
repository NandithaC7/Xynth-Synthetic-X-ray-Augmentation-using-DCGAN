type Props = {
  matrix: number[][]
  classNames: string[]
}

/** Text / CSS confusion matrix — sharp grid, blue intensity. */
export function ConfusionMatrix({ matrix, classNames }: Props) {
  const max = Math.max(...matrix.flat(), 1)

  return (
    <div className="overflow-x-auto">
      <table className="border-collapse text-sm">
        <thead>
          <tr>
            <th className="border border-border px-3 py-2 text-left text-muted">True \\ Pred</th>
            {classNames.map((name) => (
              <th key={name} className="border border-border px-3 py-2 font-medium text-dark">
                {name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={classNames[i]}>
              <th className="border border-border px-3 py-2 text-left font-medium text-dark">
                {classNames[i]}
              </th>
              {row.map((value, j) => {
                const intensity = value / max
                const bg = `rgba(30, 94, 255, ${intensity * 0.45})`
                return (
                  <td
                    key={`${i}-${j}`}
                    className="border border-border px-3 py-2 text-center font-mono"
                    style={{ backgroundColor: bg }}
                  >
                    {value}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
