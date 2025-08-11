import { motion, AnimatePresence } from "framer-motion";
import { FileText, Layers, Award, X, Code } from "lucide-react";

function AnalysisResult({
  parsedContent,
  onCareerPathClick,
  selectedCareerPath,
  isMobile,
  onClose,
}) {
  const cleanText = (text) => {
    if (!text) return text;
    return text.replace(/[^a-zA-Z0-9\s]+$/, "").trim();
  };

  return (
    <AnimatePresence>
      {parsedContent && (
        <motion.div
          initial={
            isMobile ? { opacity: 0, x: -100 } : { width: 0, opacity: 0 }
          }
          animate={
            isMobile ? { opacity: 1, x: 0 } : { width: "100%", opacity: 1 }
          }
          exit={isMobile ? { opacity: 0, x: -100 } : { width: 0, opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="bg-white shadow-xl border-r z-20 border-gray-200 flex flex-col h-full fixed top-0 left-0 w-full md:w-auto md:relative md:h-screen"
          style={{ overflowY: "hidden", maxHeight: "100vh" }}
        >
          {/* Header - Always visible */}
          <div className="p-4 md:p-6 pb-3 md:pb-4 sticky top-0 bg-white z-10">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg md:text-xl font-bold text-blue-800 flex items-center">
                <FileText className="mr-2 text-blue-600" size={20} />
                Analysis Results
              </h2>
              {onClose && (
                <button
                  onClick={onClose}
                  className="text-gray-500 hover:text-gray-700"
                  aria-label="Close sidebar"
                >
                  <X size={20} />
                </button>
              )}
            </div>
          </div>

          {/* Scrollable Content Area */}
          <div
            className="flex-1 overflow-y-auto px-4 md:px-6 pb-6"
            style={{ WebkitOverflowScrolling: "touch" }}
          >
            <div className="space-y-4 md:space-y-6">
              {/* Experiences */}
              <motion.div
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-green-50 rounded-lg p-3 md:p-4 border border-green-100"
              >
                <h2 className="text-lg md:text-xl font-bold text-green-800 mb-2 md:mb-3 flex items-center">
                  <Layers className="mr-2 text-green-600" size={20} />
                  Experiences
                </h2>
                <ul className="space-y-1 md:space-y-2">
                  {parsedContent.Experience.filter(
                    (exp, index, self) =>
                      index === self.findIndex((e) => e.title === exp.title)
                  ).map((exp, idx) => (
                    <li
                      key={idx}
                      className="text-sm md:text-base text-gray-700 pl-4 relative"
                    >
                      <span className="absolute left-0 top-2 w-2 h-2 bg-green-600 rounded-full"></span>
                      {cleanText(exp.title)}
                    </li>
                  ))}
                </ul>
              </motion.div>

              {/* Projects Section */}
              {parsedContent.Projects &&
                parsedContent.Projects.length > 0 &&
                parsedContent.Projects.some(
                  (proj) => proj.title && proj.title.trim() !== ""
                ) && (
                  <motion.div
                    initial={{ opacity: 0, x: -50 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.35 }}
                    className="bg-blue-50 rounded-lg p-3 md:p-4 border border-blue-100"
                  >
                    <h2 className="text-lg md:text-xl font-bold text-blue-800 mb-2 md:mb-3 flex items-center">
                      <Code className="mr-2 text-blue-600" size={20} />
                      Projects
                    </h2>
                    <ul className="space-y-1 md:space-y-2">
                      {(() => {
                        const validProjects = parsedContent.Projects.filter(
                          (proj, index, self) =>
                            proj.title &&
                            proj.title.trim() !== "" &&
                            index ===
                              self.findIndex((p) => p.title === proj.title)
                        );

                        const allProjectsHaveDescriptions = validProjects.every(
                          (proj) =>
                            proj.description && proj.description.trim() !== ""
                        );

                        return validProjects.map((proj, idx) => (
                          <li
                            key={idx}
                            className="text-sm md:text-base text-gray-700 pl-4 relative"
                          >
                            <span className="absolute left-0 top-2 w-2 h-2 bg-blue-600 rounded-full"></span>
                            <div>
                              <div
                                className={
                                  allProjectsHaveDescriptions
                                    ? "font-medium"
                                    : "font-normal"
                                }
                              >
                                {cleanText(proj.title)}
                              </div>
                              {allProjectsHaveDescriptions &&
                                proj.description &&
                                proj.description.trim() !== "" && (
                                  <div className="text-xs md:text-sm text-gray-600 mt-1">
                                    {cleanText(proj.description)}
                                  </div>
                                )}
                            </div>
                          </li>
                        ));
                      })()}
                    </ul>
                  </motion.div>
                )}

              {/* Career Paths */}
              <motion.div
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-yellow-50 rounded-lg p-3 md:p-4 border border-yellow-200"
              >
                <h2 className="text-lg md:text-xl font-bold text-yellow-800 mb-2 md:mb-3 flex items-center">
                  <Award className="mr-2 text-yellow-600" size={20} />
                  Career Paths
                </h2>

                <p className="text-xs md:text-sm text-gray-600 mb-2">
                  Select a career path to filter the job listings
                </p>

                <div className="space-y-3 md:space-y-4">
                  {parsedContent.jobRecommendation.map((job, idx) => {
                    const isSelected = job.jobTitle === selectedCareerPath;

                    return (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.4 + idx * 0.1 }}
                        onClick={() => {
                          onCareerPathClick(job.jobTitle);
                        }}
                        className={`bg-white shadow-sm rounded-lg p-2 md:p-3 border ${
                          isSelected
                            ? "border-blue-300 bg-blue-50"
                            : "border-gray-200"
                        } hover:cursor-pointer`}
                      >
                        <div className="flex justify-between items-center mb-1 md:mb-2">
                          <h3
                            className={`text-md md:text-md font-semibold ${
                              isSelected ? "text-blue-800" : "text-gray-800"
                            }`}
                          >
                            {cleanText(job.jobTitle)}
                          </h3>
                          {isSelected && (
                            <span className="text-xs text-blue-600">
                              Selected
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-gray-600 mb-1 md:mb-2">
                          {cleanText(job.jobDescription)}
                        </p>
                        <div className="grid grid-cols-1 gap-1 md:gap-2">
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
                              <div>
                                <h4 className="text-xs font-semibold text-gray-700 mb-1 flex items-center">
                                  Relevant Experience
                                </h4>
                                <div className="flex flex-wrap gap-1">
                                  {job.relevantExperience.map((exp, index) => {
                                    let displayText;
                                    if (typeof exp === "string") {
                                      displayText = cleanText(
                                        exp.split(":")[0].trim()
                                      );
                                    } else if (
                                      typeof exp === "object" &&
                                      exp.title
                                    ) {
                                      displayText = cleanText(exp.title);
                                    } else {
                                      displayText = cleanText(String(exp));
                                    }

                                    return (
                                      <span
                                        key={index}
                                        className="bg-purple-100 text-purple-800 px-2 py-0.5 rounded-full text-xs border border-purple-200"
                                        title={
                                          typeof exp === "object" &&
                                          exp.description
                                            ? cleanText(exp.description)
                                            : displayText
                                        }
                                      >
                                        {displayText}
                                      </span>
                                    );
                                  })}
                                </div>
                              </div>
                            )}

                            {job.relatedProjects.length > 0 && (
                              <div className="mt-2">
                                {(() => {
                                  const allProjectsHaveDescriptions =
                                    job.relatedProjects.every((proj) => {
                                      if (
                                        typeof proj === "object" &&
                                        proj.description
                                      ) {
                                        return proj.description.trim() !== "";
                                      }
                                      return false;
                                    });

                                  return (
                                    <>
                                      <h4
                                        className={`
                                          text-xs 
                                          ${
                                            allProjectsHaveDescriptions
                                              ? "font-semibold"
                                              : "font-normal"
                                          } 
                                          text-gray-700 mb-1 flex items-center
                                        `}
                                      >
                                        Related Projects
                                      </h4>
                                      <div className="flex flex-wrap gap-1">
                                        {job.relatedProjects.map(
                                          (proj, index) => {
                                            let displayText;
                                            if (typeof proj === "string") {
                                              displayText = cleanText(
                                                proj.split(":")[0].trim()
                                              );
                                            } else if (
                                              typeof proj === "object" &&
                                              proj.title
                                            ) {
                                              displayText = cleanText(
                                                proj.title
                                              );
                                            } else {
                                              displayText = cleanText(
                                                String(proj)
                                              );
                                            }

                                            return (
                                              <span
                                                key={index}
                                                className="bg-orange-100 text-orange-800 px-2 py-0.5 rounded-full text-xs border border-orange-200"
                                                title={
                                                  allProjectsHaveDescriptions &&
                                                  typeof proj === "object" &&
                                                  proj.description
                                                    ? cleanText(
                                                        proj.description
                                                      )
                                                    : displayText
                                                }
                                              >
                                                {displayText}
                                                {allProjectsHaveDescriptions &&
                                                  typeof proj === "object" &&
                                                  proj.description &&
                                                  proj.description.trim() !==
                                                    "" && (
                                                    <span className="block text-xs text-orange-600 mt-1">
                                                      {cleanText(
                                                        proj.description
                                                      )}
                                                    </span>
                                                  )}
                                              </span>
                                            );
                                          }
                                        )}
                                      </div>
                                    </>
                                  );
                                })()}
                              </div>
                            )}
                          </div>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              </motion.div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default AnalysisResult;
