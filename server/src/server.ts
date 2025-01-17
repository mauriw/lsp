/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */
import {
	createConnection,
	TextDocuments,
	Diagnostic,
	DiagnosticSeverity,
	ProposedFeatures,
	InitializeParams,
	DidChangeConfigurationNotification,
	CompletionItem,
	CompletionItemKind,
	TextDocumentPositionParams,
	TextDocumentSyncKind,
	InitializeResult
} from 'vscode-languageserver/node';

import axios from 'axios';

import { frequencyWords } from './data';

import {
	TextDocument
} from 'vscode-languageserver-textdocument';
import { strict } from 'assert';

// Create a connection for the server, using Node's IPC as a transport.
// Also include all preview / proposed LSP features.
let connection = createConnection(ProposedFeatures.all);

// Create a simple text document manager.
let documents: TextDocuments<TextDocument> = new TextDocuments(TextDocument);

let hasConfigurationCapability: boolean = false;
let hasWorkspaceFolderCapability: boolean = false;
let hasDiagnosticRelatedInformationCapability: boolean = false;

connection.onInitialize((params: InitializeParams) => {
	let capabilities = params.capabilities;

	// Does the client support the `workspace/configuration` request?
	// If not, we fall back using global settings.
	hasConfigurationCapability = !!(
		capabilities.workspace && !!capabilities.workspace.configuration
	);
	hasWorkspaceFolderCapability = !!(
		capabilities.workspace && !!capabilities.workspace.workspaceFolders
	);
	hasDiagnosticRelatedInformationCapability = !!(
		capabilities.textDocument &&
		capabilities.textDocument.publishDiagnostics &&
		capabilities.textDocument.publishDiagnostics.relatedInformation
	);

	const result: InitializeResult = {
		capabilities: {
			textDocumentSync: TextDocumentSyncKind.Incremental,
			// Tell the client that this server supports code completion.
			completionProvider: {
				resolveProvider: true,
				triggerCharacters: ['.', '(']
			}
		}
	};
	if (hasWorkspaceFolderCapability) {
		result.capabilities.workspace = {
			workspaceFolders: {
				supported: true
			}
		};
	}
	return result;
});

connection.onInitialized(() => {
	if (hasConfigurationCapability) {
		// Register for all configuration changes.
		connection.client.register(DidChangeConfigurationNotification.type, undefined);
	}
	if (hasWorkspaceFolderCapability) {
		connection.workspace.onDidChangeWorkspaceFolders(_event => {
			connection.console.log('Workspace folder change event received.');
		});
	}
});

// The example settings
interface ExampleSettings {
	maxNumberOfProblems: number;
}

// The global settings, used when the `workspace/configuration` request is not supported by the client.
// Please note that this is not the case when using this server with the client provided in this example
// but could happen with other clients.
const defaultSettings: ExampleSettings = { maxNumberOfProblems: 1000 };
let globalSettings: ExampleSettings = defaultSettings;

// Cache the settings of all open documents
let documentSettings: Map<string, Thenable<ExampleSettings>> = new Map();

connection.onDidChangeConfiguration(change => {
	if (hasConfigurationCapability) {
		// Reset all cached document settings
		documentSettings.clear();
	} else {
		globalSettings = <ExampleSettings>(
			(change.settings.languageServerExample || defaultSettings)
		);
	}

	// Revalidate all open text documents
	documents.all().forEach(validateTextDocument);
});

function getDocumentSettings(resource: string): Thenable<ExampleSettings> {
	if (!hasConfigurationCapability) {
		return Promise.resolve(globalSettings);
	}
	let result = documentSettings.get(resource);
	if (!result) {
		result = connection.workspace.getConfiguration({
			scopeUri: resource,
			section: 'languageServerExample'
		});
		documentSettings.set(resource, result);
	}
	return result;
}

// Only keep settings for open documents
documents.onDidClose(e => {
	documentSettings.delete(e.document.uri);
});

// The content of a text document has changed. This event is emitted
// when the text document first opened or when its content has changed.
documents.onDidChangeContent(change => {
	validateTextDocument(change.document);
});

async function validateTextDocument(textDocument: TextDocument): Promise<void> {
	// In this simple example we get the settings for every validate run.
	let settings = await getDocumentSettings(textDocument.uri);

	// The validator creates diagnostics for all uppercase words length 2 and more
	let text = textDocument.getText();
	let pattern = /\b[A-Z]{2,}\b/g;
	let m: RegExpExecArray | null;

	let problems = 0;
	let diagnostics: Diagnostic[] = [];
	while ((m = pattern.exec(text)) && problems < settings.maxNumberOfProblems) {
		problems++;
		let diagnostic: Diagnostic = {
			severity: DiagnosticSeverity.Warning,
			range: {
				start: textDocument.positionAt(m.index),
				end: textDocument.positionAt(m.index + m[0].length)
			},
			message: `${m[0]} is all uppercase.`,
			source: 'ex'
		};
		if (hasDiagnosticRelatedInformationCapability) {
			diagnostic.relatedInformation = [
				{
					location: {
						uri: textDocument.uri,
						range: Object.assign({}, diagnostic.range)
					},
					message: 'Spelling matters'
				},
				{
					location: {
						uri: textDocument.uri,
						range: Object.assign({}, diagnostic.range)
					},
					message: 'Particularly for names'
				}
			];
		}
		diagnostics.push(diagnostic);
	}

	// Send the computed diagnostics to VSCode.
	connection.sendDiagnostics({ uri: textDocument.uri, diagnostics });	
}

connection.onDidChangeWatchedFiles(_change => {
	// Monitored files have change in VSCode
	connection.console.log('We received an file change event');
});

// This handler provides the initial list of the completion items.
connection.onCompletion(
	(_textDocumentPosition: TextDocumentPositionParams): any => {
		// The pass parameter contains the position of the text document in
		// which code complete got requested. For the example we ignore this
		// info and always provide the same completion items.

		let document = documents.get(_textDocumentPosition.textDocument.uri);
		let text = "";
		if(document) text = document.getText();

		let pos = _textDocumentPosition.position;
		let lines = text.split('\n'); // split the text document into lines 

		let prev_line_words = Array();
		let curr_line_words = Array();

		if(lines.length > 1) prev_line_words = lines[pos.line - 1].split(' '); 
		curr_line_words = lines[pos.line].split(' ');

		let api_input = "";

		for(let i = 0; i < prev_line_words.length; i++) {
			if(i === 0) api_input = prev_line_words[i];
			else api_input += ' ' + prev_line_words[i];
		}

		for(let i = 0; i < curr_line_words.length; i++) {  
			if(i === (curr_line_words.length - 1)) {
				let curr_word = curr_line_words[curr_line_words.length - 1];
				if((curr_word.indexOf(".") != -1) || (curr_word.indexOf("(") != -1)) {
					if(api_input.length === 0) {
						api_input += curr_line_words[i];
					}
					else {
						api_input += ' ' + curr_line_words[i];
					}				
				}
			}
			else {
				if(api_input.length === 0) {
					api_input += curr_line_words[i];
				}
				else {
					api_input += ' ' + curr_line_words[i];
				}
			}
		}
		if(api_input.length === 0) api_input = "hi"; // api does not take in empty inputs
		if(api_input.charAt(api_input.length - 1) === ")") api_input = api_input.substr(0, api_input.length-1);

        return axios({
            method: 'post',
            url: 'https://api-inference.huggingface.co/models/mrm8488/CodeGPT-small-finetuned-python-token-completion',
            headers: { "Authorization": "Bearer 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" }, 
			// data: {"inputs": api_input} 
			data: {"inputs": api_input, "parameters": {"num_return_sequences": 4, "num_beams":4, "num_beam_groups":4, "diversity_penalty":0.5}, "options":{use_gpu:true}}
          }).then((response) => {
			console.log(api_input, 'input');
			// console.log(response.headers);
            let generated_text = response.data[0]['generated_text'];
			// let generated_text2 = response.data[1]['generated_text'];
			// let gt3 = response.data[2]['generated_text']
			// console.log('input', api_input);
			// console.log('gen text 1 '+generated_text);
			// console.log('gen text 2 '+generated_text2);
			// console.log('gt 3', gt3)
			// console.log(response.data[3]['generated_text'])
			let predictions = [response.data[0]['generated_text'], response.data[1]['generated_text'], response.data[2]['generated_text'], response.data[3]['generated_text']]
            //using the generated text -> 
			// let predictions = [generated_text];
			console.log(predictions);
			//console.log(predictions);
            let processed_predictions = Array();
            let aStr = 'a';
			let apiPreds = new Set();
            for(let i = 0; i < predictions.length; i++) {
                let startIndex = predictions[i].indexOf(api_input);
                if(startIndex != -1) {
                    let rest_of_string = predictions[i].substring(startIndex + api_input?.length).trim();
                    let rest_of_string_arr = rest_of_string.split(' ');
                    if(rest_of_string_arr.length > 0) {
                        aStr += 'a';
						if(!apiPreds.has(rest_of_string_arr[0])) {
							processed_predictions.push({label: rest_of_string_arr[0].replace(/\W/g, ''), sortText: aStr});
							apiPreds.add(rest_of_string_arr[0]);
						}
                    }
                }
            } 
			console.log(processed_predictions);
			// console.log(processed_predictions, predictions, input);
			for(let i = 0; i < frequencyWords.length; i++) {
				aStr = aStr + 'a';
				if(!apiPreds.has(frequencyWords[i])) processed_predictions.push({label: frequencyWords[i].replace(/\W/g, ''), sortText : aStr });
			}
			//console.log('next predictions:')
			//console.log(processed_predictions)
			//NOTE: why is import numpy as n giving the incorrect answer?
            return processed_predictions;
		})
		.catch((error) => {
			console.log(error);
		});

		// if(_textDocumentPosition.position.line === 0) result.push({label: 'hi'});
	}
);
// connection.onCompletion(
// 	(_textDocumentPosition: TextDocumentPositionParams): any => {
// 		// The pass parameter contains the position of the text document in
// 		// which code complete got requested. For the example we ignore this
// 		// info and always provide the same completion items.

// 		let document = documents.get(_textDocumentPosition.textDocument.uri);
// 		let text = "";
// 		if(document) {
// 			text = document.getText();
// 		}
// 		let pos = _textDocumentPosition.position;
// 		let lines = text.split('\n');   
// 		let input_arr = lines[pos.line].split(' ');
// 		let input = "";
// 		for(let i = 0; i < input_arr.length - 1; i++) {
// 			input += ' ' + input_arr[i];
// 		}
// 		if(input.length === 0) input = "hi";

//         return axios({
//             method: 'post',
//             url: 'https://api-inference.huggingface.co/models/mrm8488/CodeGPT-small-finetuned-python-token-completion',
//             headers: { "Authorization": "Bearer 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" }, 
//             data: {"inputs": input, "parameters": {"num_return_sequences": 4, "num_beams":4, "max_length":input.split(' ').length+1, "use_gpu":true}} 
//           }).then((response) => {
// 			console.log(response.headers);
//             let generated_text = response.data[0]['generated_text'];
// 			let generated_text2 = response.data[1]['generated_text'];
// 			let gt3 = response.data[2]['generated_text']
// 			console.log('gen text 1 '+generated_text);
// 			console.log('gen text 2 '+generated_text2);
// 			console.log('gt 3', gt3)
// 			console.log(response.data[3]['generated_text'])
// 			let predictions = [response.data[0]['generated_text'], response.data[1]['generated_text'], response.data[2]['generated_text'], response.data[3]['generated_text']]
//             //let predictions = generated_text.split('.');
// 			//console.log(predictions);
//             let processed_predictions = Array();
//             let aStr = 'a';
// 			let apiPreds = new Set();
//             for(let i = 0; i < predictions.length; i++) {
//                 let startIndex = predictions[i].indexOf(input);
//                 if(startIndex != -1) {
//                     let rest_of_string = predictions[i].substring(startIndex + input?.length).trim();
//                     let rest_of_string_arr = rest_of_string.split(' ');
//                     if(rest_of_string_arr.length > 0) {
//                         aStr += 'a';
// 						if(!apiPreds.has(rest_of_string_arr[0])) {
// 							processed_predictions.push({label: rest_of_string_arr[0], sortText: aStr});
// 							apiPreds.add(rest_of_string_arr[0]);
// 						}
//                     }
//                 }
//             } 
// 			console.log(processed_predictions);
// 			// console.log(processed_predictions, predictions, input);
// 			for(let i = 0; i < frequencyWords.length; i++) {
// 				aStr = aStr + 'a';
// 				if(!apiPreds.has(frequencyWords[i])) processed_predictions.push({label: frequencyWords[i], sortText : aStr });
// 			}
// 			//console.log('next predictions:')
// 			//console.log(processed_predictions)
// 			//NOTE: why is import numpy as n giving the incorrect answer?
//             return processed_predictions;
// 		})
// 		.catch((error) => {
// 			console.log(error);
// 		});

// 		// if(_textDocumentPosition.position.line === 0) result.push({label: 'hi'});
// 	}
// );

// This handler resolves additional information for the item selected in
// the completion list.
connection.onCompletionResolve(
	(item: CompletionItem): CompletionItem => {
		if (item.data === 1) {
			item.detail = 'TypeScript details';
			item.documentation = 'TypeScript documentation';
		} else if (item.data === 2) {
			item.detail = 'JavaScript details';
			item.documentation = 'JavaScript documentation';
		}
		return item;
	}
);

// Make the text document manager listen on the connection
// for open, change and close text document events
documents.listen(connection);

// Listen on the connection
connection.listen();
