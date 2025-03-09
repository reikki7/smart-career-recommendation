import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileText, Layers, Award } from "lucide-react";

function AnalysisResult({
  parsedContent,
  onCareerPathClick,
  selectedCareerPath,
}) {
  return (
    <AnimatePresence>
      {parsedContent && (
        <motion.div
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: "25%", opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="bg-white shadow-xl border-r border-gray-200 p-6 overflow-y-auto h-screen sticky top-0"
        >
          <div className="space-y-6">
            {/* Summary */}
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-blue-50 rounded-lg p-4 border border-blue-100"
            >
              <h2 className="text-xl font-bold text-blue-800 mb-3 flex items-center">
                <FileText className="mr-2 text-blue-600" size={24} />
                Summary
              </h2>
              <p className="text-gray-700">{parsedContent.summary}</p>
            </motion.div>

            {/* Experiences */}
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-green-50 rounded-lg p-4 border border-green-100"
            >
              <h2 className="text-xl font-bold text-green-800 mb-3 flex items-center">
                <Layers className="mr-2 text-green-600" size={24} />
                Experiences
              </h2>
              <ul className="space-y-2">
                {parsedContent.Experience
                  // Filter out duplicates by matching the first occurrence of the same title
                  .filter(
                    (exp, index, self) =>
                      index === self.findIndex((e) => e.title === exp.title)
                  )
                  .map((exp, idx) => (
                    <li key={idx} className="text-gray-700 pl-4 relative">
                      <span className="absolute left-0 top-2 w-2 h-2 bg-green-600 rounded-full"></span>
                      {exp.title}
                    </li>
                  ))}
              </ul>
            </motion.div>

            {/* Career Paths */}
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-yellow-50 rounded-lg p-4 border border-yellow-200"
            >
              <h2 className="text-xl font-bold text-yellow-800 mb-3 flex items-center">
                <Award className="mr-2 text-yellow-600" size={24} />
                Career Paths
              </h2>

              <p className="text-sm text-gray-600 mb-2">
                Select a career path to filter the job listings
              </p>

              <div className="space-y-4">
                {parsedContent.jobRecommendation.map((job, idx) => {
                  const isSelected = job.jobTitle === selectedCareerPath;

                  return (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.4 + idx * 0.1 }}
                      onClick={() => onCareerPathClick(job.jobTitle)}
                      className={`bg-white shadow-sm rounded-lg p-3 border ${
                        isSelected
                          ? "border-blue-300 bg-blue-50"
                          : "border-gray-200"
                      } hover:cursor-pointer`}
                    >
                      <div className="flex justify-between items-center mb-2">
                        <h3
                          className={`text-md font-semibold ${
                            isSelected ? "text-blue-800" : "text-gray-800"
                          }`}
                        >
                          {job.jobTitle}
                        </h3>
                        {isSelected && (
                          <span className="text-xs text-blue-600">
                            Selected
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-gray-600 mb-2">
                        {job.jobDescription}
                      </p>
                      <div className="grid grid-cols-1 gap-2">
                        <div>
                          <h4 className="text-xs font-semibold text-gray-700 mb-1">
                            Skills
                          </h4>
                          <div className="flex flex-wrap gap-1">
                            {job.skills.map((skill, index) => (
                              <span
                                key={index}
                                className="bg-green-100 text-green-800 px-2 py-0.5 rounded-full text-xs"
                              >
                                {skill}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div>
                          {job.relevantExperience.length > 0 && (
                            <h4 className="text-xs font-semibold text-gray-700 mb-1">
                              Relevant Experience
                            </h4>
                          )}
                          <div className="flex flex-wrap gap-1">
                            {job.relevantExperience.map((exp, index) => {
                              const role = exp.split(":")[0].trim();
                              return (
                                <span
                                  key={index}
                                  className="bg-purple-100 text-purple-800 px-2 py-0.5 rounded-full text-xs"
                                >
                                  {role}
                                </span>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default AnalysisResult;
