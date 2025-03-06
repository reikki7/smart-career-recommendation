import React, { useState, useEffect } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileText,
  Upload,
  ChevronRight,
  ArrowRight,
  Star,
  Briefcase,
  Layers,
  Award,
} from "lucide-react";

import ConfidenceVennDiagram from "./components/ConfidenceVennDiagram";

function App() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [pdfContent, setPdfContent] = useState(null);
  const [parsedContent, setParsedContent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [jobListings, setJobListings] = useState([]);

  // RapidAPI environment variables
  const rapidApiUrl = import.meta.env.VITE_RAPIDAPI_URL;
  const rapidApiHost = import.meta.env.VITE_RAPIDAPI_HOST;
  const rapidApiKey = import.meta.env.VITE_RAPIDAPI_KEY;

  const pdfParseEndpoint = import.meta.env.VITE_PDF_PARSING_API_URL;

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError("");
    } else {
      setFile(null);
      setFileName("");
      setError("Please select a PDF file");
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a PDF file first");
      return;
    }
    localStorage.removeItem("jobListings");
    setJobListings([]);

    setLoading(true);
    setPdfContent(null);
    setParsedContent(null);

    const formData = new FormData();
    formData.append("pdfFile", file);

    try {
      const response = await fetch(pdfParseEndpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to parse PDF");
      }

      const output = await response.text();
      const cleanedOutput = output.replace(/```/g, "");
      console.log(cleanedOutput);

      setPdfContent(cleanedOutput);
      setError("");
    } catch (err) {
      setError("Error parsing PDF: " + err.message);
      setPdfContent(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (pdfContent) {
      try {
        let cleanedJson = pdfContent.trim().replace(/```/g, "");
        if (cleanedJson.toLowerCase().startsWith("json")) {
          cleanedJson = cleanedJson.substring(4).trim();
        }
        const jsonData = JSON.parse(cleanedJson);
        setParsedContent(jsonData);
      } catch (err) {
        console.error("Error parsing JSON:", err);
      }
    }
  }, [pdfContent]);

  useEffect(() => {
    async function fetchJobListings(title) {
      const sanitizedTitle = title.replace(/[-:]/g, "");

      const options = {
        method: "GET",
        url: rapidApiUrl,
        params: {
          title_filter: sanitizedTitle,
          location_filter: "Indonesia",
        },
        headers: {
          "x-rapidapi-key": rapidApiKey,
          "x-rapidapi-host": rapidApiHost,
        },
      };
      try {
        const response = await axios.request(options);
        return response.data;
      } catch (error) {
        console.error("Error fetching job listings for", title, error);
        return [];
      }
    }

    if (parsedContent && parsedContent.jobRecommendation) {
      const cachedListings = localStorage.getItem("jobListings");
      if (cachedListings) {
        setJobListings(JSON.parse(cachedListings));
      } else {
        Promise.all(
          parsedContent.jobRecommendation.map((job) =>
            fetchJobListings(job.jobTitle)
          )
        ).then((results) => {
          const flattened = results.flat();
          setJobListings(flattened);
          // Cache the result
          localStorage.setItem("jobListings", JSON.stringify(flattened));
        });
      }
    }
  }, [parsedContent]);

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Left Sidebar (shrunk to 25%) */}
      <AnimatePresence>
        {parsedContent && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: "25%", opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="bg-white shadow-xl border-r border-gray-200 p-6 overflow-y-auto"
          >
            <div className="space-y-6">
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
                  {parsedContent.Experience.map((exp, idx) => (
                    <li key={idx} className="text-gray-700 pl-4 relative">
                      <span className="absolute left-0 top-2 w-2 h-2 bg-green-600 rounded-full"></span>
                      {exp.title}
                    </li>
                  ))}
                </ul>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-yellow-50 rounded-lg p-4 border border-yellow-200"
              >
                <h2 className="text-xl font-bold text-yellow-800 mb-3 flex items-center">
                  <Award className="mr-2 text-yellow-600" size={24} />
                  Job Recommendations
                </h2>
                <div className="space-y-4">
                  {parsedContent.jobRecommendation.map((job, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.4 + idx * 0.1 }}
                      className="bg-white shadow-sm rounded-lg p-3 border border-gray-200"
                    >
                      <div className="flex justify-between items-center mb-2">
                        <h3 className="text-md font-semibold text-gray-800">
                          {job.jobTitle}
                        </h3>
                        <span className="bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full text-xs font-medium">
                          {job.confidenceScore}% Match
                        </span>
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
                  ))}
                </div>
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        <div className="max-w-2xl mx-auto w-full px-4 py-12 space-y-6">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center"
          >
            <h1 className="text-4xl font-bold text-blue-600 mb-4 flex justify-center items-center">
              <Briefcase className="mr-3" size={36} />
              Job Recommendation
            </h1>
            <p className="text-gray-600 max-w-md mx-auto">
              Upload your resume to discover personalized job recommendations
              tailored to your skills and experience.
            </p>
          </motion.div>

          {/* File Upload */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white shadow-lg rounded-lg p-6 border border-gray-200"
          >
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload your CV
              </label>
              <div className="flex items-center">
                <input
                  type="file"
                  onChange={handleFileChange}
                  className="hidden"
                  id="pdf-upload"
                  accept="application/pdf"
                />
                <label
                  htmlFor="pdf-upload"
                  className="flex-1 cursor-pointer flex items-center py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none"
                >
                  <Upload className="mr-2" size={20} />
                  {fileName || "Choose PDF file..."}
                </label>
                <button
                  onClick={handleUpload}
                  disabled={!file || loading}
                  className="ml-3 inline-flex items-center justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none disabled:bg-blue-300 disabled:cursor-not-allowed"
                >
                  {loading ? "Processing..." : "Analyze"}
                  <ArrowRight className="ml-2" size={16} />
                </button>
              </div>
              {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
            </div>
          </motion.div>

          {/* Loading Indicator */}
          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex justify-center items-center space-x-3 text-blue-600"
            >
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-current"></div>
              <p className="text-gray-600">Analyzing your document...</p>
            </motion.div>
          )}

          {/* Confidence Pie Chart */}
          {parsedContent && (
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="bg-white shadow-lg rounded-lg p-6 border border-gray-200"
            >
              <h2 className="text-xl font-bold text-blue-600 mb-4 text-center">
                Job Match Distribution
              </h2>
              <ConfidenceVennDiagram
                jobRecommendation={parsedContent.jobRecommendation}
              />
            </motion.div>
          )}
        </div>
      </div>

      {/* Right Sidebar with Job Listings */}
      <AnimatePresence>
        {jobListings.length > 0 && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: "25%", opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="bg-white shadow-xl border-l border-gray-200 flex flex-col h-screen sticky top-0"
          >
            {/* Fixed header section */}
            <div className="p-6 pb-4">
              <motion.div
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-blue-50 rounded-lg p-4 border border-blue-100"
              >
                <h2 className="text-xl font-bold text-blue-800 mb-2 flex items-center">
                  <Briefcase className="mr-2 text-blue-600" size={24} />
                  Job Listings
                </h2>
                <p className="text-sm text-gray-700">
                  Current openings that match your profile
                </p>
              </motion.div>
            </div>

            {/* Scrollable content area */}
            <div className="flex-1 overflow-y-auto p-6 pt-0">
              <div className="space-y-4">
                {jobListings.map((job, idx) => (
                  <motion.a
                    key={idx}
                    href={job.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 + idx * 0.1 }}
                    className="block bg-white shadow-sm rounded-lg p-4 border border-gray-200 hover:shadow-md hover:border-blue-200 transition-all duration-200 cursor-pointer"
                  >
                    <div>
                      <div className="flex justify-between items-start mb-2">
                        <h3 className="text-md font-semibold text-gray-800">
                          {job.title}
                        </h3>
                        <span className="bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full text-xs font-medium">
                          {job.employment_type?.[0]?.replace("_", " ") || "Job"}
                        </span>
                      </div>

                      <p className="text-sm text-gray-600 mb-2">
                        {job.organization}
                      </p>

                      {/* Location instead of logo */}
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

                      {/* Date and Apply button in flex container */}
                      <div className="flex justify-between items-center mt-2">
                        <p className="text-xs text-gray-500">
                          Posted:{" "}
                          {new Date(job.date_posted).toLocaleDateString()}
                        </p>
                        <span className="text-blue-600 text-sm font-medium flex items-center">
                          View Details
                          <ChevronRight size={16} className="ml-1" />
                        </span>
                      </div>
                    </div>
                  </motion.a>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
