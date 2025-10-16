# RESHADED Approach for System Design

The **RESHADED** approach is a systematic framework for solving system design problems. It provides a structured methodology to ensure all critical aspects of a system are considered during the design process.

## Overview

While there is no universal solution for all system design challenges, RESHADED offers consistent guidance through key steps that help create comprehensive, well-thought-out designs.

## The Seven Parts

### **R - Requirements**
**"Gather all the requirements of the design problem and define its scope."**

This is the foundation of any system design. It involves:
- **Functional Requirements**: What the system should do
  - Core features and capabilities
  - User interactions and use cases
  - Business logic and workflows

- **Non-Functional Requirements**: How the system should behave
  - Performance (latency, throughput)
  - Scalability (handling growth)
  - Availability (uptime targets)
  - Reliability (fault tolerance)
  - Security (authentication, authorization)
  - Consistency guarantees

**Purpose**: Clearly define what you're building and establish boundaries for the design scope.

---

### **E - Estimation**
**"Calculate infrastructure needs for a specified user base."**

Back-of-the-envelope calculations to determine resource requirements:
- **User Load**: Daily/monthly active users, concurrent users
- **Traffic**: Requests per second (read/write ratio)
- **Storage**: Data volume, growth rate, retention period
- **Bandwidth**: Network throughput requirements
- **Compute**: Server count, CPU/memory per server

**Example Questions**:
- How many servers are needed for 500 million daily active users?
- How much storage for one billion photos?
- What bandwidth for 10,000 requests per second?

**Purpose**: Quantify the scale and understand infrastructure needs before diving into design.

---

### **S - Storage Schema**
**"Define tables and field types for data modeling."**

An optional but important step for data-centric systems:
- Database schema design
- Table structures and relationships
- Field types and constraints
- Indexes and keys
- Data partitioning strategy

**When to Include**:
- Systems with complex data models
- Applications requiring persistent storage
- Designs involving multiple entities and relationships

**Purpose**: Model the data layer to support functional requirements efficiently.

---

### **H - High-level Design**
**"Identify main components and building blocks."**

Create the initial architecture diagram showing:
- Major system components (services, databases, caches)
- Communication patterns between components
- Dataflow through the system
- Load balancers, queues, and middleware
- External services and integrations

**Inspired By**: Functional and non-functional requirements

**Purpose**: Provide a bird's-eye view of the system architecture before diving into details.

---

### **A - API Design**
**"Build interfaces for our service through API calls."**

Translate functional requirements into concrete API endpoints:
- RESTful endpoints or RPC methods
- Request/response formats
- Authentication and authorization mechanisms
- Rate limiting and throttling
- Versioning strategy

**Example**:
```
POST /users - Create user
GET /users/{id} - Get user details
POST /posts - Create post
GET /feed - Get user feed
```

**Purpose**: Define how clients interact with the system and establish clear contracts.

---

### **D - Detailed Design**
**"Recognize high-level limitations and evolve the design."**

Refine the high-level design by addressing:
- Component internals and implementation details
- Technology choices (databases, caches, message queues)
- Workflow details and edge cases
- Error handling and retry mechanisms
- Monitoring and observability
- Addressing bottlenecks and single points of failure

**Focus Areas**:
- How does each component work internally?
- What happens when components fail?
- How do we handle the peak load?
- What are the data consistency guarantees?

**Purpose**: Finalize all aspects of the design to meet both functional and non-functional requirements.

---

### **E - Evaluation**
**"Measure solution effectiveness and discuss trade-offs."**

Critically assess the design:
- How does it fulfill functional requirements?
- Does it meet non-functional requirements (scalability, availability)?
- What are the trade-offs made?
  - CAP theorem implications (Consistency vs. Availability)
  - Cost vs. performance
  - Complexity vs. maintainability
- What are the limitations?
- What could fail and how is it handled?

**Purpose**: Justify design decisions and demonstrate understanding of trade-offs.

---

### **D - Distinctive Component/Feature**
**"Address unique aspects specific to each problem."**

Focus on what makes this system special or challenging:

**Examples**:
- **Uber**: Payment processing, driver-rider matching, real-time location tracking
- **Google Docs**: Real-time collaborative editing, operational transforms, conflict resolution
- **Netflix**: Content delivery network (CDN), recommendation engine, adaptive bitrate streaming
- **Twitter**: Feed generation, trending topics, handling celebrity tweets (hotspot data)
- **WhatsApp**: End-to-end encryption, message delivery guarantees, presence indicators

**Purpose**: Demonstrate a deep understanding of the problem's unique challenges and how to solve them.

---

## Key Advantages

1. **Systematic Approach**: Provides clear next steps throughout the design process
2. **Comprehensive Coverage**: Ensures all necessary aspects are considered
3. **Structured Thinking**: Helps organize thoughts and communicate clearly
4. **Interview-Ready**: Maps well to system design interview expectations
5. **Flexibility**: Can adapt depth based on time constraints and problem complexity

## Usage Tips

- **Time Management**: Allocate time proportionally (Requirements 10%, Estimation 5%, High-level 25%, Detailed 40%, Evaluation 15%, Distinctive 5%)
- **Interviewer Collaboration**: Clarify requirements early and check assumptions
- **Iterate**: Start simple, then add complexity based on requirements
- **Justify Decisions**: Always explain *why* you chose a particular approach
- **Know Trade-offs**: Be prepared to discuss alternatives and their implications

## Example Application

**Problem**: Design a URL shortener like bit.ly

**R - Requirements**:
- Functional: Create short URL, redirect to original URL, custom aliases
- Non-functional: Low latency (<10 ms), high availability (99.99%), handle 1B URLs

**E - Estimation**:
- 100M URLs created per month → ~40 URLs/sec
- Read:Write ratio 100:1 → 4000 reads/sec
- Storage: 1B URLs × 500 bytes = 500 GB

**S - Storage Schema**:
```
URL Table: id, short_code, original_url, user_id, created_at, expires_at
```

**H - High-level Design**:
- Load Balancer → Application Servers → Cache → Database
- Key generation service for unique short codes

**A - API Design**:
```
POST /shorten - Create short URL
GET /{short_code} - Redirect to original URL
```

**D - Detailed Design**:
- Base62 encoding for short codes
- Redis cache for hot URLs
- Cassandra for distributed storage
- Consistent hashing for partitioning

**E - Evaluation**:
- Meets latency via caching
- Handles scale via horizontal scaling
- Trade-off: Eventual consistency for availability

**D - Distinctive**:
- Unique short code generation (avoiding collisions)
- Handling expired URLs
- Analytics (click tracking)

---

## Conclusion

RESHADED provides a robust framework for tackling system design problems methodically. By following this approach, 
you ensure comprehensive coverage of all aspects while maintaining a clear structure and logical flow. Remember: adapt 
the depth of each component based on the problem's complexity and time available.