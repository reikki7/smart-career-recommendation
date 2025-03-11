import { useState, useEffect, useRef } from "react";
import axios from "axios";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";

import ManualInput from "./components/ManualInput";
import ResumeUpload from "./components/ResumeUpload";
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
  const [inputMode, setInputMode] = useState("upload");
  const [isSticky, setIsSticky] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    {
      role: "assistant",
      content: "I can answer questions about your results. How can I help you?",
    },
  ]);
  const [manualFormData, setManualFormData] = useState({
    name: "",
    education: [{ school: "", degree: "", major: "", gpa: "" }],
    thesisAbstract: "",
    experience: [{ title: "", description: "" }],
    projects: [{ title: "", description: "" }],
    languages: [""],
    softSkills: [""],
  });

  const tabHeaderRef = useRef(null);

  const backendEndpoint =
    import.meta.env.VITE_BACKEND_API_URL || "http://localhost:5000";

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
      const response = await axios.post(
        `${backendEndpoint}/api/parse-pdf`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
            skip_zrok_interstitial: true,
          },
        }
      );

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
          console.error("Error parsing response:", jsonError);
          const cleanedOutput = response.data.replace(/```/g, "");
          console.log("Cleaned non-JSON output:", cleanedOutput);
          setPdfContent(cleanedOutput);
        }
      }

      setError("");
    } catch (err) {
      console.error("Error during PDF upload:", err);
      const errorMessage = err.response?.data || err.message;
      setError("Error parsing document: " + errorMessage);
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
    const handleScroll = () => {
      if (tabHeaderRef.current) {
        const headerPos = tabHeaderRef.current.getBoundingClientRect().top;
        setIsSticky(headerPos <= 0);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  useEffect(() => {
    async function fetchJobListings(title) {
      const sanitizedTitle = title.replace(/[-:]/g, "");
      try {
        const response = await axios.get(
          `${backendEndpoint}/api/linkedin-jobs`,
          {
            headers: {
              "Content-Type": "application/json",
              "ngrok-skip-browser-warning": true,
              skip_zrok_interstitial: true,
            },
            params: { title: sanitizedTitle },
          }
        );
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
  }, [parsedContent, backendEndpoint, isLoadingJobs]);

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

      {/* Main Content Section */}
      <div className="flex-1 flex flex-col">
        {isSticky && <div className="h-12"></div>}
        {/* Tab Header */}
        <div
          ref={tabHeaderRef}
          className={`flex justify-center space-x-4 border-b border-gray-200 bg-gray-50 z-0 w-full ${
            isSticky ? "fixed top-0 left-0 right-0" : ""
          }`}
        >
          <button
            onClick={() => {
              setInputMode("upload");
            }}
            className={`py-2 px-4 font-semibold ${
              inputMode === "upload"
                ? "border-b-2 border-blue-600 text-blue-600"
                : "text-gray-600"
            }`}
          >
            Resume Upload
          </button>
          <button
            onClick={() => {
              setInputMode("manual");
            }}
            className={`py-2 px-4 font-semibold ${
              inputMode === "manual"
                ? "border-b-2 border-blue-600 text-blue-600"
                : "text-gray-600"
            }`}
          >
            Manual Input
          </button>
        </div>

        {/* Render the appropriate input form */}
        {inputMode === "upload" ? (
          <ResumeUpload
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
            chatMessages={chatMessages}
            setChatMessages={setChatMessages}
          />
        ) : (
          <ManualInput
            backendEndpoint={backendEndpoint}
            setParsedContent={setParsedContent}
            parsedContent={parsedContent}
            setIsLoadingJobs={setIsLoadingJobs}
            setJobListings={setJobListings}
            setSelectedCareerPath={setSelectedCareerPath}
            setIsAnalyzing={setIsAnalyzing}
            chatMessages={chatMessages}
            setChatMessages={setChatMessages}
            formData={manualFormData}
            setFormData={setManualFormData}
          />
        )}
      </div>

      {/* Right Sidebar with job listings */}
      <JobListings
        jobListings={jobListings}
        selectedCareerPath={selectedCareerPath}
        setSelectedCareerPath={setSelectedCareerPath}
      />
    </div>
  );
}

export default App;
