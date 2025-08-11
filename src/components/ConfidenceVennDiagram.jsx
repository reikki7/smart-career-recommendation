import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { motion } from "framer-motion";
import { memo, useMemo } from "react";

const ConfidenceVennDiagram = ({ jobRecommendation }) => {
  const confidenceData = useMemo(() => {
    if (!jobRecommendation || jobRecommendation.length === 0) {
      return [];
    }

    const totalConfidenceScore = jobRecommendation.reduce(
      (sum, job) => sum + (parseFloat(job.confidence) || 0),
      0
    );

    return jobRecommendation.map((job) => ({
      name: job.jobTitle,
      value: parseFloat(job.confidence) || 0,
      similarityScore: parseFloat(job.similarity_score) || 0,
      percentage:
        totalConfidenceScore > 0
          ? ((parseFloat(job.confidence) / totalConfidenceScore) * 100).toFixed(
              1
            )
          : "0.0",
    }));
  }, [jobRecommendation]);

  const COLORS = [
    "#0088FE",
    "#00C49F",
    "#FFBB28",
    "#FF8042",
    "#8884D8",
    "#82ca9d",
    "#ffc658",
  ];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const color = payload[0].color;

      return (
        <motion.div
          initial={{ opacity: 0, scale: 0.9, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
          className="bg-white/95 backdrop-blur-sm shadow-xl rounded-xl p-5 border border-gray-200/80 max-w-xs min-w-[200px]"
          style={{
            boxShadow:
              "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
          }}
        >
          {/* Header with colored indicator */}
          <div className="flex items-start gap-3 mb-3">
            <div
              className="w-4 h-4 rounded-full flex-shrink-0 mt-0.5 ring-2 ring-white shadow-sm"
              style={{ backgroundColor: color }}
            />
            <h4 className="font-semibold text-gray-900 leading-tight text-sm">
              {data.name}
            </h4>
          </div>

          {/* Divider */}
          <div className="border-t border-gray-100 mb-3" />

          {/* Stats */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-gray-600 text-xs font-medium">
                % of Total
              </span>
              <span className="text-gray-900 font-semibold text-sm">
                {data.percentage}%
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-gray-600 text-xs font-medium">
                Similarity
              </span>
              <span className="text-blue-600 font-semibold text-sm">
                {(data.similarityScore * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Combined progress bar showing all segments */}
          <div className="mt-4">
            <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
              <div className="flex h-full">
                {confidenceData.map((item, index) => (
                  <div
                    key={index}
                    className={`h-full transition-all duration-300 ${
                      item.name === data.name
                        ? "opacity-100 ring-1 ring-white"
                        : "opacity-30"
                    }`}
                    style={{
                      width: `${item.percentage}%`,
                      backgroundColor: COLORS[index % COLORS.length],
                    }}
                  />
                ))}
              </div>
            </div>

            {/* Similarity score progress bar */}
            <div className="w-full bg-gray-100 rounded-full h-1.5 mt-2">
              <div
                className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                style={{
                  width: `${(data.similarityScore * 100).toFixed(1)}%`,
                }}
              />
            </div>
          </div>
        </motion.div>
      );
    }
    return null;
  };

  const CustomLegend = (props) => {
    const { payload } = props;
    return (
      <div className="flex flex-wrap justify-center gap-2 mt-4">
        {payload.map((entry, index) => (
          <div key={index} className="flex items-center gap-1 text-xs">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-gray-700 max-w-[120px] truncate">
              {entry.value}
            </span>
          </div>
        ))}
      </div>
    );
  };

  if (!confidenceData || confidenceData.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex items-center justify-center h-64 text-gray-500"
      >
        No recommendation data available
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="flex flex-col justify-center items-center bg-white rounded-lg p-6 shadow-sm border border-gray-200"
    >
      <h3 className="text-lg font-semibold text-gray-800 mb-4 text-center">
        Career Match Confidence Distribution
      </h3>
      <div className="w-full max-w-md sm:max-w-lg md:max-w-xl lg:max-w-2xl">
        <ResponsiveContainer width="100%" height={400}>
          <PieChart>
            <Pie
              data={confidenceData}
              cx="50%"
              cy="50%"
              labelLine={false}
              outerRadius={120}
              fill="#8884d8"
              dataKey="value"
              animationBegin={0}
              animationDuration={800}
            >
              {confidenceData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                  className="focus:outline-none duration-150 hover:opacity-70 cursor-pointer"
                />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend content={<CustomLegend />} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
};

export default memo(ConfidenceVennDiagram);
