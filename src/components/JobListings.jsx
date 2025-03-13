import { motion, AnimatePresence } from "framer-motion";
import { ChevronRight, X, Briefcase } from "lucide-react";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import linkedinIcon from "../assets/Linkedin_icon.svg";

dayjs.extend(relativeTime);

function JobListings({
  jobListings,
  selectedCareerPath,
  setSelectedCareerPath,
  isMobile,
  onClose,
}) {
  const filteredJobs = selectedCareerPath
    ? jobListings.filter((job) => job.careerPath === selectedCareerPath)
    : jobListings;

  const showEmptyState = selectedCareerPath || jobListings.length > 0;

  return (
    <AnimatePresence>
      {showEmptyState && (
        <motion.div
          initial={isMobile ? { opacity: 0, x: 100 } : { width: 0, opacity: 0 }}
          animate={
            isMobile ? { opacity: 1, x: 0 } : { width: "100%", opacity: 1 }
          }
          exit={isMobile ? { opacity: 0, x: 100 } : { width: 0, opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="bg-white shadow-xl border-l z-20 border-gray-200 flex flex-col h-full fixed top-0 right-0 w-full md:w-auto md:relative md:h-screen"
          style={{ overflowY: "hidden", maxHeight: "100vh" }}
        >
          {/* Header */}
          <div className="p-4 md:p-6 pb-3 md:pb-4 sticky top-0 bg-white z-10">
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-blue-50 rounded-lg p-3 md:p-4 border border-blue-100"
            >
              {/* Header */}
              <div className="flex justify-between items-center mb-2">
                <h2 className="text-lg md:text-xl font-bold text-blue-800 flex items-center">
                  {isMobile ? (
                    <img
                      src={linkedinIcon}
                      alt="LinkedIn"
                      className="w-5 h-5 md:w-6 md:h-6 mr-2"
                    />
                  ) : (
                    <img
                      src={linkedinIcon}
                      alt="LinkedIn"
                      className="w-5 h-5 md:w-6 md:h-6 mr-2"
                    />
                  )}
                  Job Listings
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

              <p className="text-xs md:text-sm text-gray-700">
                Current openings that match your profile
              </p>

              {selectedCareerPath && (
                <p className="text-xs text-gray-600 mt-1 md:mt-2">
                  Showing jobs for <strong>{selectedCareerPath}</strong>
                </p>
              )}
            </motion.div>
          </div>

          {/* Content Area */}
          <div
            className="flex-1 overflow-y-auto p-4 md:p-6 pt-0"
            style={{ WebkitOverflowScrolling: "touch" }}
          >
            <AnimatePresence mode="wait">
              <motion.div
                key={selectedCareerPath || "all-jobs"}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.3 }}
              >
                {filteredJobs.length > 0 ? (
                  <div className="space-y-3 md:space-y-4 pb-4 md:pb-0">
                    {filteredJobs.map((job, idx) => (
                      <motion.a
                        key={idx}
                        href={job.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.3, delay: idx * 0.05 }}
                        className="block bg-white shadow-sm rounded-lg p-3 md:p-4 border border-gray-200 hover:shadow-md hover:border-blue-200 transition-all duration-200 cursor-pointer"
                      >
                        <div>
                          {job.companyLogo && (
                            <img
                              src={job.companyLogo}
                              alt={job.organization}
                              className="w-10 h-10 md:w-12 md:h-12 object-contain mb-2 md:mb-3"
                            />
                          )}
                          <div className="flex justify-between items-start mb-1 md:mb-2">
                            <h3 className="text-sm md:text-md font-semibold text-gray-800">
                              {job.title}
                            </h3>
                          </div>
                          <p className="text-xs md:text-sm text-gray-600 mb-1 md:mb-2">
                            {job.organization}
                          </p>

                          {/* Career Path Tag */}
                          {job.careerPath && (
                            <div className="mb-1 md:mb-2">
                              <span className="inline-block bg-indigo-100 text-indigo-800 text-xs font-medium px-2 py-0.5 rounded-full text-xs">
                                {job.careerPath}
                              </span>
                            </div>
                          )}

                          {/* Location */}
                          <div className="flex items-center text-gray-600 text-xs md:text-sm mb-2 md:mb-3">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="h-3 w-3 md:h-4 md:w-4 mr-1"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                              />
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                              />
                            </svg>
                            {job.locations_derived?.[0] || "Remote"}
                          </div>

                          {/* Date */}
                          <div
                            className={`flex items-center mt-1 md:mt-2 ${
                              job.date_posted
                                ? "justify-between"
                                : "justify-end"
                            }`}
                          >
                            {job.date_posted &&
                            !isNaN(
                              new Date(job.date_posted + "T00:00:00Z").getTime()
                            ) ? (
                              <p className="text-xs text-gray-500">
                                {dayjs(job.date_posted).format("MMM D, YYYY")}
                                {` — ${dayjs(job.date_posted).fromNow()}`}
                              </p>
                            ) : null}

                            <span className="text-blue-600 text-xs md:text-sm font-medium flex items-center">
                              View Details
                              <ChevronRight size={14} className="ml-1" />
                            </span>
                          </div>
                        </div>
                      </motion.a>
                    ))}
                  </div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex flex-col items-center justify-center py-6 md:py-8 px-4 text-center"
                  >
                    <div className="bg-gray-50 rounded-full p-3 md:p-4 mb-2 md:mb-3">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-8 w-8 md:h-10 md:w-10 text-gray-400"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={1.5}
                          d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                    </div>
                    <h3 className="text-sm md:text-base text-gray-700 font-medium mb-1 md:mb-2">
                      No relevant jobs found
                    </h3>
                    <p className="text-xs md:text-sm text-gray-500">
                      {selectedCareerPath
                        ? `We couldn't find any jobs matching "${selectedCareerPath}" at this time.`
                        : "We couldn't find any jobs matching your profile at this time."}
                    </p>
                    {selectedCareerPath && (
                      <button
                        onClick={() => {
                          setSelectedCareerPath("");
                          if (isMobile && onClose) {
                            onClose();
                          }
                        }}
                        className="mt-3 md:mt-4 text-blue-600 text-xs md:text-sm font-medium"
                      >
                        View all available positions
                      </button>
                    )}
                  </motion.div>
                )}
              </motion.div>
            </AnimatePresence>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default JobListings;
