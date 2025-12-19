import { inlineCopilot } from "./codemirror-copilot/index.ts";

// --- AI COPILOT INTEGRATION ---
function createCopilotExtension(function_info) {
    const settingManager = SettingManager.getInstance();
    const aiSettings = settingManager.get('ai');

    if (!aiSettings || !aiSettings.enabled) {
        return []; // Return an empty array if disabled
    }
    //let functions = function_info.map(f => `- ${f.value.split('\n')[0].substring(4)}`).join('\n');
    let functions = function_info.map(f =>{
        let l=f.value.split('\n');
        return `- ${l[0].substring(4)} | `+l.slice(1).join(' | ');
    }).join('\n');
    // Generate the concise context prompt
    const contextPrompt = `Code is run in the web workder context, you can use the following global libraries:
- math.js available as 'math'
- d3.js available as 'd3'
- SheetJS available as 'XLSX'
- bfjs available as 'bfjs' (a big number library for **floating point** javascript, usage: let a=bfjs.bf(string/number);let b=bfjs.sin(number);let c=a+b*12; c.toString(radix); c.toBigInt(); c.toNumber(); etc. Do NOT mix up bfjs BigFloat with native BigInt or Number types, use conversions explicitly)
- browser web worker APIs (like fetch, import, importScripts etc. DOM api is also available, use document.body.append if you need, but echoHTML is suggested.) 
- build in browser APIs (like console, BigInt, URL, URLSearchParams, TextEncoder, TextDecoder, btoa, atob etc.)
All the code runs in an async function context, so you can use 'await' directly. Functions like readFile, writeFile must be awaited.
Available functions:
${functions}
`;
    //console.log("AI Copilot context prompt:", contextPrompt);

    let onSuggestionRequest=async (prefix, suffix) => {
        const aiSettings = settingManager.get('ai');
        const { apiUrl, apiKey, model, enabled, advMode } = aiSettings;

        let systemPrompt;
        if(advMode){
            systemPrompt = 
`You are an expert JavaScript programmer. Your task is to analyze a snippet of code and provide a completion or a replacement.
Current cursor position is marked with <!--CUR_CURSOR-->. 
**WARNING:** Reply with DIRECT json content. Do **NOT** add any explanation or markdown formatting.
The result is directly parsed by JSON.parse, any markdown formatting elements will lead to failure.

You **MUST** respond in a structured JSON format with the following fields:
- text (string): The code to insert at the cursor's position.
- linesToDelete (number): The number of lines to delete (from the current cursor's line).

Example response for the prompt:
//write Hello, World! to the console
console.log(<!--CUR_CURSOR-->'an useless line.');
console.log('Goodbye!');

Your response should be:
{
  "text": "console.log('Hello, World!');",
  "linesToDelete": 2,
}

linesToDelete = 2 means to delete the current line and the next line.
**WARNING** Compare the generated code with the existing code and only keep the necessary lines.
**ALWAYS** remove duplicate or redundant lines of code if your completion is an overwrite to the existing code.
If linesToDelete >= 1, it means the current line is deleted. And you should repeat the code on the current line if needed.
If linesToDelete >= 1, it means the current line is deleted. And you should repeat the code on the current line if needed.
If linesToDelete >= 1, it means the current line is deleted. And you should repeat the code on the current line if needed.
`;
        }else{
            systemPrompt = 
`You are an expert JavaScript programmer. Your task is to analyze a snippet of code and provide a completion.
Only output the code that replaces <!--CUR_CURSOR--> part. 
**WARNING:** Do **NOT** add any explanation or MARKDOWN around the code.
The returned result is directly inserted into the javascript code, any markdown formatting elements will lead to failure.

`;
        }

        if(!enabled) {
            return;
        }
        if (!apiUrl) {
            console.warn("AI Copilot is enabled, but the API URL is not set.");
            return [];
        }
        
        const userPrompt = `${prefix}<!--CUR_CURSOR-->${suffix}`;
        const messages = [
            { role: 'system', content: systemPrompt + '\n' + contextPrompt },
            { role: 'user', content: userPrompt }
        ];
        
        try {
            let requestJson = JSON.stringify({
                    model: model,
                    messages: messages,
                    stream: false,
                    temperature: 0,
                });
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`
                },
                body: requestJson
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`API request failed with status ${response.status}: ${errorText}`);
            }

            const data = await response.json();
            let prediction;
            if(advMode){
                prediction = JSON.parse(data.choices[0]?.message?.content);
            }else{
                prediction = {'text':data.choices[0]?.message?.content, linesToDelete:0 };
            }
            if (prediction) {
                const oldUsage = settingManager.get('ai.usage');
                let promptTokens = 0;
                let completionTokens = 0;

                if (data.usage && data.usage.prompt_tokens) {
                    // Use token data from API response if available
                    promptTokens = data.usage.prompt_tokens;
                    completionTokens = data.usage.completion_tokens;
                } else {
                    // Fallback to character count if token data is not available
                    promptTokens = requestJson.length;
                    completionTokens = prediction.text.length;
                }
                
                settingManager.set('ai.usage', {
                    prompt_tokens: (oldUsage.prompt_tokens || 0) + promptTokens,
                    completion_tokens: (oldUsage.completion_tokens || 0) + completionTokens,
                });
            }

            return prediction;
        } catch (error) {
            console.error('Error calling AI Copilot API:', error);
            return ''; // Return empty string on error to prevent breaking the editor
        }
    };
    return inlineCopilot({onSuggestionRequest,delay:0,acceptOnClick:true,hotkey:"Alt-i",maxPrefix:10000,maxSuffix:10000});
}
export { createCopilotExtension };