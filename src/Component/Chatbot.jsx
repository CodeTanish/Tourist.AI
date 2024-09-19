import React, { useState } from "react";
import { LuSendHorizonal } from "react-icons/lu"; // Corrected the icon name
import { FaUser, FaRobot } from "react-icons/fa"; // User and bot icons
import './style.css';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [userMessage, setUserMessage] = useState("");
  const [showDateSelection, setShowDateSelection] = useState(false);
  const [showPassengerInfo, setShowPassengerInfo] = useState(false);
  const [passengerInfo, setPassengerInfo] = useState({
    adults: 1,
    children: 0
  });

  const handleSendMessage = async () => {
    if (userMessage.trim() === "") return;

    // Add the user's message to the chat
    const userMessageObj = { from: "user", text: userMessage };
    setMessages((prevMessages) => [...prevMessages, userMessageObj]);

    // Send the message to the Flask API
    const response = await fetch("https://cb-ui-51lj.onrender.com/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMessage }),
    });

    const data = await response.json();
    const botMessageObj = { from: "bot", text: data.response };
    
    // Add the bot's response to the chat
    setMessages((prevMessages) => [...prevMessages, botMessageObj]);

    // Check for flags in the response
    if (data.showDateSelection) {
      setShowDateSelection(true);
      setShowPassengerInfo(false);
    } else if (data.showPassengerInfo) {
      setShowPassengerInfo(true);
      setShowDateSelection(false);
    } else {
      // Hide both forms if neither flag is set
      setShowDateSelection(false);
      setShowPassengerInfo(false);
    }

    setUserMessage("");
  };

  // Function to handle date selection
  const handleDateSelection = async (date) => {
    const dateMessageObj = { from: "user", text: date };
    setMessages((prevMessages) => [...prevMessages, dateMessageObj]);
    
    // Send the selected date to the chatbot
    const response = await fetch("http://localhost:5000/select-date", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ date })
    });
    
    const data = await response.json();
    const botMessageObj = { from: "bot", text: data.response };
    setMessages((prevMessages) => [...prevMessages, botMessageObj]);

    setShowDateSelection(false); // Hide date selection after selection
  };

  // Function to handle passenger info submission
  const handlePassengerInfoSubmit = async () => {
    const passengerMessage = `Adults: ${passengerInfo.adults}, Children: ${passengerInfo.children}`;
    const passengerMessageObj = { from: "user", text: passengerMessage };
    setMessages((prevMessages) => [...prevMessages, passengerMessageObj]);

    // Send the passenger info to the chatbot
    const response = await fetch("http://localhost:5000/submit-passenger-info", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(passengerInfo),
    });
    
    const data = await response.json();
    const botMessageObj = { from: "bot", text: data.response };
    setMessages((prevMessages) => [...prevMessages, botMessageObj]);

    setShowPassengerInfo(false); // Hide passenger info form after submission
  };

  return (
    <div className="chat-container">
      <h1 className="head">Chat with AI</h1>
      <div className="chat-box">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.from}`}>
            {msg.from === "user" ? (
              <FaUser className="icon user-icon" />
            ) : (
              <FaRobot className="icon bot-icon" />
            )}
            <div className="message-text">{msg.text}</div>
          </div>
        ))}
      </div>

      {/* Date Selection */}
      {showDateSelection && (
        <div className="date-selection">
          <h3>Select a Date:</h3>
          <button onClick={() => handleDateSelection("Today")}>Today</button>
          <button onClick={() => handleDateSelection("Tomorrow")}>Tomorrow</button>
        </div>
      )}

      {/* Passenger Info */}
      {showPassengerInfo && (
        <div className="passenger-info">
          <h3>Enter the number of adults and children:</h3>
          <label>
            Adults:
            <input
              type="number"
              value={passengerInfo.adults}
              onChange={(e) => setPassengerInfo({ ...passengerInfo, adults: e.target.value })}
            />
          </label>
          <label>
            Children:
            <input
              type="number"
              value={passengerInfo.children}
              onChange={(e) => setPassengerInfo({ ...passengerInfo, children: e.target.value })}
            />
          </label>
          <button onClick={handlePassengerInfoSubmit}>Submit</button>
        </div>
      )}

      {/* Input Area */}
      {!showDateSelection && !showPassengerInfo && (
        <div className="input-area">
          <input
            type="text"
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
            placeholder="Type your message..."
          />
          <button className="m-btn" onClick={handleSendMessage}>
            <LuSendHorizonal className="btn" />
          </button>
        </div>
      )}
    </div>
  );
};

export default Chatbot;
