import { useState, useEffect } from "react";
import axios from "axios";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";

import MainContent from "./components/MainContent";
import JobListings from "./components/JobListings";
import AnalysisResult from "./components/AnalysisResult";

dayjs.extend(relativeTime);

function App() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [pdfContent, setPdfContent] = useState(null);
  const [parsedContent, setParsedContent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [jobListings, setJobListings] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isLoadingJobs, setIsLoadingJobs] = useState(false);
  const [vennDiagramRendered, setVennDiagramRendered] = useState(false);
  const [selectedCareerPath, setSelectedCareerPath] = useState("");

  const pdfParseEndpoint = import.meta.env.VITE_PDF_PARSING_API_URL;
  const linkedinJobsEndpoint = import.meta.env.VITE_LINKEDIN_JOBS_API_URL;

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
    setSelectedCareerPath("");

    setIsAnalyzing(true);
    setLoading(true);
    setIsLoadingJobs(false);
    setPdfContent(null);
    setParsedContent(null);

    const formData = new FormData();
    formData.append("pdfFile", file);

    try {
      const response = await axios.post(pdfParseEndpoint, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (typeof response.data === "object") {
        console.log("Received valid JSON response:", response.data);
        setParsedContent(response.data);
        setIsLoadingJobs(true);
      } else {
        try {
          const jsonData = JSON.parse(response.data);
          setParsedContent(jsonData);
          setIsLoadingJobs(true);
        } catch (jsonError) {
          console.error("Error parsing response as JSON:", jsonError);
          const cleanedOutput = response.data.replace(/```/g, "");
          console.log("Cleaned non-JSON output:", cleanedOutput);
          setPdfContent(cleanedOutput);
        }
      }

      setError("");
    } catch (err) {
      console.error("Error during PDF upload:", err);
      const errorMessage = err.response?.data || err.message;
      setError("Error parsing PDF: " + errorMessage);
      setPdfContent(null);
      setParsedContent(null);
      setIsLoadingJobs(false);
    } finally {
      setIsAnalyzing(false);
      setLoading(false);
    }
  };

  useEffect(() => {
    if (parsedContent) {
      setVennDiagramRendered(false);
      setIsLoadingJobs(true);
    } else if (pdfContent) {
      try {
        const cleanedJson = pdfContent
          .trim()
          .replace(/```/g, "")
          .replace(/^json\s*:?/i, "");
        const jsonEndIndex = cleanedJson.lastIndexOf("}");
        if (jsonEndIndex !== -1) {
          const jsonPart = cleanedJson.substring(0, jsonEndIndex + 1);
          const jsonData = JSON.parse(jsonPart);
          setParsedContent(jsonData);
          setVennDiagramRendered(false);
          setIsLoadingJobs(true);
        } else {
          throw new Error("Could not find valid JSON structure");
        }
      } catch (err) {
        console.error("Error parsing JSON:", err);
        setError(
          "We couldn't parse the data from your PDF. Please try again or upload a different PDF."
        );
        setParsedContent(null);
        setIsLoadingJobs(false);
      }
    }
  }, [pdfContent, parsedContent]);

  useEffect(() => {
    async function fetchJobListings(title) {
      const sanitizedTitle = title.replace(/[-:]/g, "");
      try {
        const response = await axios.get(linkedinJobsEndpoint, {
          params: { title: sanitizedTitle },
        });
        console.log("Fetched job listings for", title, response.data);

        const jobsWithCareerPath = response.data.map((job) => ({
          ...job,
          careerPath: title,
        }));

        return jobsWithCareerPath;
      } catch (error) {
        console.error("Error fetching job listings for", title, error);
        return [];
      }
    }

    if (parsedContent?.jobRecommendation?.length && isLoadingJobs) {
      const cachedListings = localStorage.getItem("jobListings");
      if (cachedListings) {
        setJobListings(JSON.parse(cachedListings));
        setIsLoadingJobs(false);
        setVennDiagramRendered(true);
      } else {
        const uniqueTitles = [
          ...new Set(
            parsedContent.jobRecommendation.map((job) => job.jobTitle)
          ),
        ];

        Promise.all(uniqueTitles.map(fetchJobListings)).then((results) => {
          const flattened = results.flat();
          setJobListings(flattened);
          localStorage.setItem("jobListings", JSON.stringify(flattened));
          setIsLoadingJobs(false);
          setVennDiagramRendered(true);
        });
      }
    }
  }, [parsedContent, linkedinJobsEndpoint, isLoadingJobs]);

  const handleCareerPathClick = (careerPath) => {
    if (selectedCareerPath === careerPath) {
      setSelectedCareerPath("");
    } else {
      setSelectedCareerPath(careerPath);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Left Sidebar with the analysis results */}
      <AnalysisResult
        parsedContent={parsedContent}
        onCareerPathClick={handleCareerPathClick}
        selectedCareerPath={selectedCareerPath}
      />

      {/* Main Content */}
      <MainContent
        fileName={fileName}
        error={error}
        loading={loading}
        isAnalyzing={isAnalyzing}
        isLoadingJobs={isLoadingJobs}
        parsedContent={parsedContent}
        vennDiagramRendered={vennDiagramRendered}
        setVennDiagramRendered={setVennDiagramRendered}
        handleFileChange={handleFileChange}
        handleUpload={handleUpload}
      />

      {/* Right Sidebar with job listings */}
      <JobListings
        jobListings={jobListings}
        selectedCareerPath={selectedCareerPath}
      />
    </div>
  );
}

export default App;
