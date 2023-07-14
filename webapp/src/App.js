import './App.css';
import axios from  'axios';
import { useState } from "react";

import styles from "@chatscope/chat-ui-kit-styles/dist/default/styles.min.css";
import {
  MainContainer,
  ChatContainer,
  MessageList,
  Message,
  MessageInput,
  TypingIndicator
} from "@chatscope/chat-ui-kit-react";

const QUERY_URL = "/api/query";

const Base64Image = ({ base64Data }) => {
	const imageUrl = `data:image/png;base64,${base64Data}`;
  
	return <img src={imageUrl} alt="Chat plot" />;
  };

function App() {
	const [loading, setLoading] = useState(false);
	const [messages, setMessages] = useState([]);

	const handleSend = async (_, text) => {
		setLoading(true);

		messages.push({
			text,
			isImage: false,
			time: new Date(),
			sender: "Robert",
			direction: "outgoing"
		});

		try {
			const encodedText = encodeURIComponent(text);
			const url = `${QUERY_URL}?q=${encodedText}`;

			// Send the request using Axios or any other HTTP client library
			const response = await axios.get(url);

			messages.push({
				text: response.data.data,
				isImage: response.data.is_image,
				time: new Date(),
				sender: "Griz",
				direction: "incoming"
			});
			setMessages(messages);

			setLoading(false);
		} catch (error) {
			console.error('Error sending the request:', error);
		}
	}

  return (
    <div className="App">
			<h2>
				Ask a question about one of these tables:
			</h2>
			<ul>
				<li>bitcoin_blockchain blocks</li>
				<li>bitcoin_blockchain transactions</li>
				<li>austin_waste waste_and_diversion</li>
				<li>austin_bikeshare bikeshare_stations</li>
				<li>austin_bikeshare bikeshare_trips</li>
				<li>census_bureau_construction new_residential_construction</li>
				<li>census_opportunity_atlas tract_covariates</li>
				<li>chicago_crime crime</li>
				<li>covid19_aha hospital_beds</li>
				<li>covid19_aha staffing2</li>
			</ul>
			<MainContainer className="main-container">
				<ChatContainer>
					<MessageList typingIndicator={loading ? <TypingIndicator content="Griz is typing" /> : null}>
						{messages.map((m, idx) => {
							console.log(m);
							if (m.isImage) {
								return (
									<Message.ImageContent
										src={`data:image/png;base64,${m.text}`}
										alt="Requested chart"
										width={320}
									/>
								);
							} else {
								return (
									<Message
									key={idx}
									model={{
										message: m.text,
										direction: m.direction,
										sentTime: `${m.time}`,
										sender: m.sender,
									}}
								/>
								);
							}
						})}
					</MessageList>
					<MessageInput attachButton={false} onSend={handleSend} placeholder="Type prompt here" />
				</ChatContainer>
			</MainContainer>
		</div>
  );
}

export default App;
