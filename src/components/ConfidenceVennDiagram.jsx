import { PieChart, Pie, Cell, Tooltip, Legend } from "recharts";
import { motion } from "framer-motion";
import { memo, useMemo } from "react";

const ConfidenceVennDiagram = ({ jobRecommendation }) => {
  const confidenceData = useMemo(() => {
    const totalConfidenceScore = jobRecommendation.reduce(
      (sum, job) => sum + parseFloat(job.confidenceScore),
      0
    );

    return jobRecommendation.map((job) => ({
      name: job.jobTitle,
      value: parseFloat(job.confidenceScore),
      percentage: (
        (parseFloat(job.confidenceScore) / totalConfidenceScore) *
        100
      ).toFixed(2),
    }));
  }, [jobRecommendation]);

  const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884D8"];

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
          <p className="text-gray-600">{data.percentage}%</p>
        </motion.div>
      );
    }
    return null;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="flex flex-col justify-center items-center"
    >
      <PieChart width={500} height={450}>
        <Pie
          data={confidenceData}
          cx={250}
          cy={180}
          labelLine={false}
          outerRadius={160}
          fill="#8884d8"
          dataKey="value"
        >
          {confidenceData.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={COLORS[index % COLORS.length]}
              className="focus:outline-none duration-150 hover:opacity-70"
            />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
        <Legend
          layout="horizontal"
          verticalAlign="bottom"
          align="center"
          wrapperStyle={{ paddingTop: "20px" }}
        />
      </PieChart>
    </motion.div>
  );
};

export default memo(ConfidenceVennDiagram);
