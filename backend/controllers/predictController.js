const axios = require("axios");
const fs = require("fs");
const FormData = require("form-data");

exports.handlePrediction = async (req, res) => {
  try {
    console.log("👉 Request received");

    if (!req.file) {
      console.log("❌ No file received");
      return res.status(400).json({ error: "No file uploaded" });
    }

    const filePath = req.file.path;
    console.log("📁 File path:", filePath);

    const formData = new FormData();
    formData.append("file", fs.createReadStream(filePath));

    console.log("🚀 Sending to ML service...");

    const response = await axios.post(
      "http://127.0.0.1:8001/predict",
      formData,
      {
        headers: formData.getHeaders(),
        timeout: 10000 // prevent hanging forever
      }
    );

    console.log("✅ ML response received");

    fs.unlinkSync(filePath);

    return res.json(response.data);

  } catch (error) {
    console.error("❌ ERROR:", error.message);
    return res.status(500).json({ error: "Prediction failed" });
  }
};


// const axios = require("axios");
// const fs = require("fs");
// const FormData = require("form-data");

// exports.handlePrediction = async (req, res) => {
//   try {
//     const filePath = req.file.path;

//     const formData = new FormData();
//     formData.append("file", fs.createReadStream(filePath));

//     // 🔥 call ML API
//     const response = await axios.post(
//       "http://127.0.0.1:8001/predict",
//       formData,
//       {
//         headers: formData.getHeaders(),
//       }
//     );

//     // delete temp file
//     fs.unlinkSync(filePath);

//     res.json(response.data);

//   } catch (error) {
//     console.error(error);
//     res.status(500).json({ error: "Prediction failed" });
//   }
// };