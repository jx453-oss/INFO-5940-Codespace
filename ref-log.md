# Reflection Log - Assignment 2: Multi-Agent Travel Planner

## What I Learned from Implementing a Multi-Agent Workflow

This assignment showed me how specialized agents working together can produce better results than a single general-purpose agent. The separation between the Planner (creative, fast, no internet) and Reviewer (fact-checking with live search) creates a nice balance between ideation and validation. It mirrors how I actually plan trips - first brainstorm, then verify details online.

The biggest lesson was about prompt engineering. You have to be extremely explicit. I had to tell the Planner "provide actual restaurant names, not generic suggestions" and tell the Reviewer "you MUST use the search tool multiple times." Without clear boundaries, agents don't stay in their lanes.

## Challenges Faced and How I Addressed Them

Getting the Planner to be specific enough was hard. My first prompt generated vague suggestions like "visit museums." I fixed this by requiring actual venue names, time slots, and costs in the output structure.

The Reviewer initially wouldn't use the search tool - it just made comments without verifying facts. I solved this by structuring the output format to require "Search Finding" fields for each validation point, forcing it to actually search before responding.

There was also a frustrating API key setup issue. I got a 401 error because the .env file still had placeholder text instead of my actual keys. Once I updated OPENAI_API_KEY and TAVILY_API_KEY with real credentials, everything worked.

## Creative Ideas and Design Choices

I designed the Reviewer's output with a "Delta List" showing specific issues and fixes before the final itinerary. This transparency helps users understand what was corrected and why, rather than just seeing silent edits.

I also added a "Positive Validations" section so the Reviewer acknowledges what the Planner got right, making feedback feel more balanced.

For both agents, I emphasized practical constraints like realistic travel times and opening hours to avoid generating itineraries that look great but are physically impossible to execute.

## External Tools and GenAI Assistance Used

I used Claude as my primary development assistant to help write and refine the agent prompts. It was particularly helpful when I was stuck on how to phrase requirements clearly and when debugging issues like getting the Reviewer to actually use the search tool. Claude helped me think through edge cases and structure the prompts more effectively.

