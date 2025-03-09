import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronRight } from "lucide-react";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import linkedinIcon from "../assets/Linkedin_icon.svg";

dayjs.extend(relativeTime);

function JobListings({ jobListings, selectedCareerPath }) {
  const filteredJobs = selectedCareerPath
    ? jobListings.filter((job) => job.careerPath === selectedCareerPath)
    : jobListings;

  return (
    <AnimatePresence>
      {jobListings.length > 0 && (
        <motion.div
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: "25%", opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="bg-white shadow-xl border-l border-gray-200 flex flex-col h-screen sticky top-0"
        >
          {/* Header */}
          <div className="p-6 pb-4">
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.05 }}
              className="bg-blue-50 rounded-lg p-4 border border-blue-100"
            >
              <h2 className="text-xl font-bold text-blue-800 mb-2 flex items-center">
                <img
                  src={linkedinIcon}
                  alt="LinkedIn"
                  className="w-6 h-6 mr-2"
                />
                Job Listings
              </h2>
              <p className="text-sm text-gray-700">
                Current openings that match your profile
              </p>

              {selectedCareerPath && (
                <p className="text-xs text-gray-600 mt-2">
                  Showing jobs for <strong>{selectedCareerPath}</strong>
                </p>
              )}
            </motion.div>
          </div>

          {/* Content Area */}
          <div className="flex-1 overflow-y-auto p-6 pt-0">
            <AnimatePresence mode="wait">
              <motion.div
                key={selectedCareerPath || "all-jobs"}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.05 }}
              >
                {filteredJobs.length > 0 ? (
                  <div className="space-y-4">
                    {filteredJobs.map((job, idx) => (
                      <motion.a
                        key={idx}
                        href={job.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.05, delay: idx * 0.02 }}
                        className="block bg-white shadow-sm rounded-lg p-4 border border-gray-200 hover:shadow-md hover:border-blue-200 transition-all duration-200 cursor-pointer"
                      >
                        <div>
                          {job.companyLogo && (
                            <img
                              src={job.companyLogo}
                              alt={job.organization}
                              className="w-12 h-12 object-contain mb-3"
                            />
                          )}
                          <div className="flex justify-between items-start mb-2">
                            <h3 className="text-md font-semibold text-gray-800">
                              {job.title}
                            </h3>
                          </div>
                          <p className="text-sm text-gray-600 mb-2">
                            {job.organization}
                          </p>

                          {/* Career Path Tag */}
                          {job.careerPath && (
                            <div className="mb-2">
                              <span className="inline-block bg-indigo-100 text-indigo-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
                                {job.careerPath}
                              </span>
                            </div>
                          )}

                          {/* Location */}
                          <div className="flex items-center text-gray-600 text-sm mb-3">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="h-4 w-4 mr-1"
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
                            className={`flex items-center mt-2 ${
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

                            <span className="text-blue-600 text-sm font-medium flex items-center">
                              View Details
                              <ChevronRight size={16} className="ml-1" />
                            </span>
                          </div>
                        </div>
                      </motion.a>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">
                    No jobs found for that career path.
                  </p>
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
