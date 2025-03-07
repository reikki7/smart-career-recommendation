import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, ArrowRight, Briefcase } from "lucide-react";
import ConfidenceVennDiagram from "./ConfidenceVennDiagram";

function MainContent({
  fileName,
  error,
  loading,
  isAnalyzing,
  isLoadingJobs,
  parsedContent,
  vennDiagramRendered,
  setVennDiagramRendered,
  handleFileChange,
  handleUpload,
}) {
  return (
    <div
      className={`flex-1 flex flex-col duration-150 ${
        !parsedContent ? "justify-center items-center" : ""
      }`}
    >
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
                disabled={loading || isAnalyzing}
                className="hidden"
                id="pdf-upload"
                accept="application/pdf"
              />
              <label
                htmlFor="pdf-upload"
                className="flex-1 cursor-pointer flex items-center py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white duration-150 hover:bg-gray-50 focus:outline-none"
              >
                <Upload className="mr-2" size={20} />
                {fileName || "Choose PDF file..."}
              </label>
              <button
                onClick={handleUpload}
                disabled={(!fileName && !isAnalyzing) || loading || isAnalyzing}
                className="ml-3 inline-flex items-center duration-150 justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none disabled:bg-blue-300 disabled:cursor-not-allowed"
              >
                {loading || isAnalyzing ? "Processing..." : "Analyze"}
                <ArrowRight className="ml-2" size={16} />
              </button>
            </div>
            {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
          </div>
        </motion.div>

        {/* Venn Diagram */}
        {parsedContent && !vennDiagramRendered && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            onAnimationComplete={() => setVennDiagramRendered(true)}
            className="bg-white shadow-lg rounded-lg p-6 border border-gray-200 mt-4"
          >
            <h2 className="text-xl font-bold text-blue-600 mb-4 text-center">
              Job Match Distribution
            </h2>
            <ConfidenceVennDiagram
              jobRecommendation={parsedContent.jobRecommendation}
            />
          </motion.div>
        )}

        {parsedContent && vennDiagramRendered && (
          <div className="bg-white shadow-lg rounded-lg p-6 border border-gray-200 mt-4">
            <h2 className="text-xl font-bold text-blue-600 mb-4 text-center">
              Job Match Distribution
            </h2>
            <ConfidenceVennDiagram
              jobRecommendation={parsedContent.jobRecommendation}
            />
          </div>
        )}

        {/* Loading Animation */}
        <AnimatePresence mode="wait">
          {(loading || isLoadingJobs) && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex justify-center items-center space-x-3 mt-4 text-blue-600"
            >
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-current"></div>
              <p className="text-gray-600">
                {isAnalyzing
                  ? "Analyzing your document..."
                  : isLoadingJobs
                  ? "Seeking job applications..."
                  : "Processing..."}
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

export default MainContent;
