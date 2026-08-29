import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useStore, activeSeries } from "../store";
import { COLORS, LABELS, type SeriesKey } from "../types";

const AXIS = { stroke: "#6e7681", fontSize: 11 };

function useRows(field: "energy" | "q") {
  const trajectory = useStore((s) => s.trajectory);
  const mode = useStore((s) => s.mode);

  return useMemo(() => {
    if (!trajectory) return [];
    return trajectory.t.map((t, i) => {
      const row: Record<string, number> = { t };
      for (const k of ["truth", "mlp", "hnn", "node"] as SeriesKey[]) {
        const src =
          mode === "naive" && k !== "truth" ? trajectory.naive : trajectory.contact;
        const v = src[k]?.[field][i];
        if (v !== undefined) row[k] = v;
      }
      return row;
    });
  }, [trajectory, mode, field]);
}

/** Playhead time, quantised so the charts re-render ~12x/s instead of 60x/s.
 *  Recharts only renders ReferenceLine as a direct child of the chart, so this
 *  has to be a value the chart itself subscribes to, not a wrapper component. */
const CURSOR_STRIDE = 5;

function useCursorTime(): number | null {
  return useStore((s) => {
    if (!s.trajectory) return null;
    const i = Math.floor(s.frame / CURSOR_STRIDE) * CURSOR_STRIDE;
    return s.trajectory.t[Math.min(i, s.trajectory.t.length - 1)];
  });
}

function Chart({
  rows,
  title,
  unit,
  domain,
}: {
  rows: Record<string, number>[];
  title: string;
  unit: string;
  domain?: [number, number];
}) {
  const visible = useStore((s) => s.visible);
  const keys = activeSeries(visible);
  const cursor = useCursorTime();

  return (
    <div className="chart">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={190}>
        <LineChart data={rows} margin={{ top: 4, right: 12, bottom: 4, left: -8 }}>
          <CartesianGrid stroke="#21262d" />
          <XAxis
            dataKey="t"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v: number) => v.toFixed(1)}
            {...AXIS}
          />
          <YAxis
            {...AXIS}
            domain={domain ?? ["auto", "auto"]}
            tickFormatter={(v: number) => v.toFixed(1)}
          />
          <Tooltip
            contentStyle={{
              background: "#161b22",
              border: "1px solid #30363d",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelFormatter={(v) => `t = ${Number(v).toFixed(2)} s`}
            formatter={(v, name) => [
              `${Number(v).toFixed(3)} ${unit}`,
              LABELS[name as SeriesKey] ?? String(name),
            ]}
          />
          {cursor !== null && (
            <ReferenceLine x={cursor} stroke="#8b949e" strokeDasharray="3 3" />
          )}
          {keys.map((k) => (
            <Line
              key={k}
              type="monotone"
              dataKey={k}
              stroke={COLORS[k]}
              strokeWidth={k === "truth" ? 2.4 : 1.6}
              strokeDasharray={k === "hnn" ? "6 3" : k === "node" ? "2 3" : undefined}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Charts() {
  const energy = useRows("energy");
  const height = useRows("q");
  return (
    <>
      <Chart rows={energy} title="Total energy H(q, p)" unit="J" />
      <Chart rows={height} title="Height q(t)" unit="m" domain={[-2.5, 2.5]} />
    </>
  );
}
