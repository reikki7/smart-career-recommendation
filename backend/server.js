const express = require("express");
const multer = require("multer");
const cors = require("cors");
const pdfParse = require("pdf-parse");
const fs = require("fs");
const path = require("path");
const axios = require("axios");

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, "uploads");
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, uniqueSuffix + "-" + file.originalname);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (file.mimetype === "application/pdf") {
      cb(null, true);
    } else {
      cb(new Error("Only PDF files are allowed"), false);
    }
  },
});

app.post("/api/parse-pdf", upload.single("pdfFile"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).send("No file uploaded");

    const pdfBuffer = fs.readFileSync(req.file.path);
    const data = await pdfParse(pdfBuffer);
    fs.unlinkSync(req.file.path);

    const lmPayload = {
      model: "local-model",
      messages: [
        {
          role: "system",
          content: `Use the following PDF document for RAG purposes:\n\n${data.text}`,
        },
        {
          role: "user",
          content: `Please summarize who the person, and list some job recommendations. The job recommendations should be based on the document content and follow a VERY specific format which will be used for further processing in the frontend. For Skills, it's the related skills to the job based on the document and the recommended job, so if you want to, for example, recommend Machine Learning Engineer, make sure I have the relevant experience from the document and also list only related skills in the JSON, like don't list PHP when the job recommendation is 3D Game Development, or Java for Machine Learning, like who uses Java for Machine Learning. And also in the JSON, for the skills, only list skillset and not the project, like if a job recommendation is Machine Learning, the skill would be like "Python" or "Tensorflow" and not "AI Chatbot", Relevant Projects/Experience will instead have its own array, it doesn't have to be more than 1 if the person only have 1 relevant project/experience, but make sure the ones listed are the exact same as the one in "Experience" array.\n\n
          
          Btw, Experience Title is the project/experience name, like what project the person was working on, or what the person did, like say it "Designer Portfolio Website" and NOT "Full-Stack Development", it should be specific and not what the project/experience is categorized in, also not "Teaching Assistant: Taught structural analysis techniques relevant to geotech engineering.", "Teaching Assistant" is enough, make it short. Make sure "relevantExperience" in jobRecommendation only lists the available experience from "Experience".\n\n
          
          For confidenceScore, it's like the points it is compared to the other job recommendations. Like for example, let's say my resume is mostly about Full Stack Development, then the confidence score for a Full Stack Developer job recommendation would be like 90, but for example, if there's a bit about Data Science, then the score would be like 30. The confidence score is between 0-100. The job recommendations should be in order of relevance.\n\n       
IMPORTANT: Return only valid JSON without any additional text or commentary.
        
JSON Format:
{
  "summary": "[summary in a paragraph, not bullet points]",
  "Experience": [
    {
      "title": "[title of the experience1]",
      "description": "[description of the experience1]"
    },
    {
      "title": "[title of the experience2]",
      "description": "[description of the experience2]"
    },
    {
      "title": "[title of the experience3]",
      "description": "[description of the experience3]"
    },
    {
    (and so on... if there are more, show more, if there are less, show less)
    }
  ],
  "jobRecommendation": [
    {
      "jobTitle": "[jobTitle1]" (String),
      "jobDescription": "[jobDescription1] (String)",
      "skills": "[skills1, skills1, skills1, ...] (Array of Strings)",
      "relevantExperience": "[relevantExperience1, relevantExperience1, relevantExperience1, ...] (Array of Strings)",
      "confidenceScore": "[confidenceScore1 (from 0-100)] (Number)"
    },
    {
      "jobTitle": "[jobTitle2]" (String),
      "jobDescription": "[jobDescription2] (String)",
      "skills": "[skills2, skills2, skills2, ...] (Array of Strings)",
      "relevantExperience": "[relevantExperience2, relevantExperience2, relevantExperience2, ...] (Array of Strings)",
      "confidenceScore": "[confidenceScore2 (from 0-100)] (Number)"
    },
    {
      "jobTitle": "[jobTitle3]" (String),
      "jobDescription": "[jobDescription3] (String)",
      "skills": "[skills3, skills3, skills3, ...] (Array of Strings)",
      "relevantExperience": "[relevantExperience3, relevantExperience3, relevantExperience3, ...] (Array of Strings)",
      "confidenceScore": "[confidenceScore3 (from 0-100)] (Number)"
    }
      {
  (and so on... if there are more, show more, if there are less, show less)
      }
  ]
}`,
        },
      ],
      temperature: 0.7,
    };

    const lmResponse = await axios.post(
      "http://localhost:1234/v1/chat/completions",
      lmPayload
    );

    const lmFormatted = lmResponse.data.choices[0].message.content;
    const cleanedContent = lmFormatted
      .replace(/<think>[\s\S]*?<\/think>/gi, "")
      .trim();

    // Build a markdown formatted output
    const formattedOutput = `
### Document Analysis

**Title:** ${data.info?.Title || "Untitled"}  
**Author:** ${data.info?.Author || "Unknown"}  
**Total Pages:** ${data.numpages}  
**Creation Date:** ${data.info?.CreationDate || "N/A"}  
**Producer:** ${data.info?.Producer || "N/A"}

### Document Summary

${cleanedContent}
`;

    res.setHeader("Content-Type", "text/html");
    res.send(cleanedContent);
  } catch (error) {
    console.error("Error processing PDF and LM Studio request:", error);
    res.status(500).send("Failed to process PDF or LM Studio request");
  }
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
