"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

import type { GlucosePoint } from "../lib/glucose";

type Props = {
  data: GlucosePoint[];
};

function formatTime(timestamp: string) {
  return new Date(timestamp).toLocaleTimeString(
    "id-ID",
    {
      hour: "2-digit",
      minute: "2-digit",
    },
  );
}

export default function GlucoseChart({
  data,
}: Props) {
  const chartData = data.map((item) => ({
    ...item,
    time: formatTime(item.timestamp),
  }));

  return (
    <div className="glucose-chart">
      <div className="chart-header">
        <div>
          <p className="eyebrow">
            Glucose history
          </p>

          <h2>
            Recent glucose trend
          </h2>
        </div>

        <span className="chart-meta">
          Last {data.length} measurements
        </span>
      </div>

      <div className="chart-container">
        {data.length === 0 ? (
          <div className="chart-empty">
            No glucose data available.
          </div>
        ) : (
          <ResponsiveContainer
            width="100%"
            height={320}
          >
            <LineChart
              data={chartData}
              margin={{
                top: 10,
                right: 20,
                left: 0,
                bottom: 10,
              }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                vertical={false}
              />

              <XAxis
                dataKey="time"
                tick={{
                  fontSize: 12,
                }}
                tickLine={false}
                axisLine={false}
              />

              <YAxis
                domain={["auto", "auto"]}
                tick={{
                  fontSize: 12,
                }}
                tickLine={false}
                axisLine={false}
                width={45}
              />

              <Tooltip
                formatter={(value) => [
                  `${Number(value).toFixed(1)} mg/dL`,
                  "Glucose",
                ]}
                labelFormatter={(label) =>
                  `Time: ${label}`
                }
              />

              <Line
                type="monotone"
                dataKey="glucose"
                strokeWidth={3}
                dot={{
                  r: 3,
                }}
                activeDot={{
                  r: 6,
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}