import { PieChart, Pie, Cell, Tooltip, Legend } from "recharts";
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
      return (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white shadow-lg rounded-lg p-4 border border-gray-200"
        >
          <p className="font-bold text-gray-800">{data.name}</p>
          <p className="text-gray-600">{data.value.toFixed(1)}%</p>
          <p className="text-gray-500 text-sm">
            Similarity: {(data.similarityScore * 100).toFixed(1)}%
          </p>
          <p className="text-blue-600 font-medium">
            {data.percentage}% of total
          </p>
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
      <PieChart width={400} height={320}>
        <Pie
          data={confidenceData}
          cx={200}
          cy={140}
          labelLine={false}
          outerRadius={100}
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
    </motion.div>
  );
};

export default memo(ConfidenceVennDiagram);
