import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);

      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleUpload = () => {
    if (!selectedFile) {
      setMessage('Please select a file first');
      return;
    }

    setLoading(true);
    setMessage('');

    const reader = new FileReader();
    reader.readAsDataURL(selectedFile);

    reader.onloadend = async () => {
      try {
        const base64Image = reader.result;

        const response = await axios.post('http://localhost:5000/upload', {
          image: base64Image,
          filename: selectedFile.name,
        });
        console.log(response.data);

        setMessage(response.data.gemini_analysis);
      } catch (error) {
        console.error('Error uploading image:', error);
        setMessage('Error uploading image. Please try again.');
      } finally {
        setLoading(false);
      }
    };
  };

  return (
    <div className="image-upload-container">
      <h2>Upload Image</h2>

      <div className="upload-controls">
        <input
          type="file"
          onChange={handleFileChange}
          className="file-input"
        />
        <button
          onClick={handleUpload}
          disabled={loading}
          className="upload-button"
        >
          {loading ? 'Uploading...' : 'Upload Image'}
        </button>
      </div>

      {previewUrl && (
        <div className="preview-container">
          <h4>Preview:</h4>
          <img src={previewUrl} alt="Preview" className="image-preview" />
        </div>
      )}

      {message && (
        <div className="message">{message}</div>
      )}

      <style>
        {`
          .image-upload-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            font-family: Arial, sans-serif;
          }

          .upload-controls {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 20px;
          }

          .file-input {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
          }

          .upload-button {
            padding: 10px 15px;
            background-color: #4285f4;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
          }

          .upload-button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
          }

          .preview-container {
            margin-top: 20px;
          }

          .image-preview {
            max-width: 100%;
            max-height: 300px;
            border-radius: 4px;
            border: 1px solid #ddd;
          }

          .message {
            margin-top: 20px;
            padding: 10px;
            border-radius: 4px;
            background-color: #e3f2fd;
          }
        `}
      </style>
    </div>
  );
}

export default App;
