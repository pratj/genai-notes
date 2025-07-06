# Complete System Design & Architecture Notes

## Table of Contents
1. [Scalability Fundamentals](#scalability-fundamentals)
2. [Load Balancing](#load-balancing)
3. [Database Systems](#database-systems)
4. [Caching Strategies](#caching-strategies)
5. [Microservices vs Monolith](#microservices-vs-monolith)
6. [Message Queues & Event Streaming](#message-queues--event-streaming)
7. [API Design Patterns](#api-design-patterns)
8. [Content Delivery Networks (CDN)](#content-delivery-networks-cdn)
9. [Distributed Systems Concepts](#distributed-systems-concepts)
10. [System Monitoring & Observability](#system-monitoring--observability)
11. [Security Architecture](#security-architecture)
12. [Common System Design Patterns](#common-system-design-patterns)

---

## Scalability Fundamentals

### Vertical vs Horizontal Scaling

| Aspect | Vertical Scaling (Scale Up) | Horizontal Scaling (Scale Out) |
|--------|---------------------------|--------------------------------|
| **Definition** | Adding more power to existing server | Adding more servers to the pool |
| **Cost** | Expensive for high-end hardware | More cost-effective for large scale |
| **Complexity** | Simple - no code changes needed | Complex - requires distributed design |
| **Limits** | Hardware limits (single point) | Theoretically unlimited |
| **Downtime** | Required for upgrades | Zero downtime with proper design |
| **Failure Impact** | Single point of failure | Better fault tolerance |

**When to use Vertical Scaling:**
- Simple applications with predictable load
- Legacy systems that can't be easily distributed
- When you need quick performance improvements
- Database systems that benefit from more RAM/CPU

**When to use Horizontal Scaling:**
- High-traffic web applications
- Systems requiring high availability
- Applications with unpredictable load patterns
- Modern distributed architectures

### Key Scalability Metrics
- **Throughput**: Requests per second the system can handle
- **Latency**: Time to process a single request
- **Availability**: Percentage of time system is operational
- **Consistency**: How synchronized data is across nodes

---

## Load Balancing

### Load Balancing Algorithms

| Algorithm | Description | Use Case | Pros | Cons |
|-----------|-------------|----------|------|------|
| **Round Robin** | Requests distributed sequentially | Equal server capacity | Simple, fair distribution | Doesn't consider server load |
| **Weighted Round Robin** | Servers assigned weights based on capacity | Servers with different specs | Accounts for server differences | Static weights |
| **Least Connections** | Routes to server with fewest active connections | Long-running connections | Dynamic load consideration | Overhead of tracking connections |
| **IP Hash** | Uses client IP to determine server | Session affinity needed | Consistent routing | Uneven distribution possible |
| **Geographic** | Routes based on user location | Global applications | Reduced latency | Complex setup |

### Layer 4 vs Layer 7 Load Balancing

| Feature | Layer 4 (Transport) | Layer 7 (Application) |
|---------|-------------------|----------------------|
| **Decision Based On** | IP address and port | HTTP headers, URLs, cookies |
| **Performance** | Faster, less processing | Slower, more processing |
| **Features** | Basic routing | Content-based routing, SSL termination |
| **Protocol Support** | TCP, UDP | HTTP, HTTPS, WebSocket |
| **Use Cases** | High-performance scenarios | Complex routing requirements |

**Layer 4 Example**: AWS Network Load Balancer
**Layer 7 Example**: AWS Application Load Balancer, NGINX

### Load Balancer Placement Strategies
1. **Internet-facing**: Between users and web servers
2. **Internal**: Between web servers and application servers
3. **Database**: Between application servers and databases

---

## Database Systems

### SQL vs NoSQL Comparison

| Aspect | SQL (Relational) | NoSQL (Non-Relational) |
|--------|------------------|------------------------|
| **Data Structure** | Structured, tables with rows/columns | Flexible: documents, key-value, graph, column-family |
| **Schema** | Fixed schema, predefined | Dynamic schema, schema-less |
| **ACID Properties** | Full ACID compliance | Eventual consistency, BASE properties |
| **Scaling** | Vertical scaling primarily | Horizontal scaling designed |
| **Query Language** | SQL (standardized) | Varies by database type |
| **Consistency** | Strong consistency | Eventual consistency |

### Database Types Deep Dive

#### 1. Relational Databases (RDBMS)
**Examples**: PostgreSQL, MySQL, Oracle, SQL Server

**Strengths:**
- ACID compliance ensures data integrity
- Complex queries with JOINs
- Mature ecosystem and tools
- Strong consistency

**Weaknesses:**
- Difficult to scale horizontally
- Fixed schema can be restrictive
- Can be slower for simple operations

**Use Cases:**
- Financial systems requiring ACID properties
- Complex business applications with relationships
- Applications requiring complex queries

#### 2. Document Databases
**Examples**: MongoDB, CouchDB, Amazon DocumentDB

**Strengths:**
- Flexible schema for evolving data
- Natural fit for object-oriented programming
- Easy to scale horizontally
- Good performance for simple queries

**Weaknesses:**
- Limited query capabilities compared to SQL
- Potential for data duplication
- Eventual consistency issues

**Use Cases:**
- Content management systems
- User profiles and preferences
- Product catalogs

#### 3. Key-Value Stores
**Examples**: Redis, DynamoDB, Riak

**Strengths:**
- Extremely fast for simple operations
- Easy to scale horizontally
- Simple data model
- High availability

**Weaknesses:**
- Limited query capabilities
- No complex relationships
- Simple data structure only

**Use Cases:**
- Caching layers
- Session storage
- Shopping carts
- Real-time recommendations

#### 4. Graph Databases
**Examples**: Neo4j, Amazon Neptune, ArangoDB

**Strengths:**
- Excellent for relationship-heavy data
- Fast traversal of connections
- Natural representation of networks
- Complex relationship queries

**Weaknesses:**
- Specialized use cases only
- Learning curve for graph query languages
- Limited ecosystem compared to SQL

**Use Cases:**
- Social networks
- Recommendation engines
- Fraud detection
- Network analysis

### Database Partitioning Strategies

#### Horizontal Partitioning (Sharding)
**Sharding Strategies:**

| Strategy | Description | Pros | Cons | Use Case |
|----------|-------------|------|------|----------|
| **Range-based** | Partition by value ranges | Simple, range queries efficient | Hot spots possible | Time-series data |
| **Hash-based** | Partition by hash function | Even distribution | No range queries | User data |
| **Directory-based** | Lookup service for partitions | Flexible, easy rebalancing | Additional complexity | Complex sharding needs |

#### Vertical Partitioning
- Split tables by columns
- Separate frequently accessed columns
- Useful for wide tables with mixed access patterns

### Database Replication Patterns

| Pattern | Description | Pros | Cons | Use Case |
|---------|-------------|------|------|----------|
| **Master-Slave** | One write node, multiple read nodes | Read scaling, backup | Write bottleneck | Read-heavy applications |
| **Master-Master** | Multiple write nodes | Write scaling, no single point | Conflict resolution needed | Distributed writes |
| **Peer-to-Peer** | All nodes equal | High availability | Complex consistency | Distributed systems |

---

## Caching Strategies

### Cache Patterns

#### 1. Cache-Aside (Lazy Loading)
```
Application -> Check Cache -> If miss -> Database -> Update Cache
```
**Pros:** Cache only contains requested data, resilient to cache failures
**Cons:** Cache miss penalty, stale data possible
**Use Case:** Read-heavy applications

#### 2. Write-Through
```
Application -> Cache -> Database (synchronously)
```
**Pros:** Data always consistent, no cache misses
**Cons:** Write latency, cache pollution
**Use Case:** Applications requiring consistency

#### 3. Write-Behind (Write-Back)
```
Application -> Cache (immediate) -> Database (asynchronously)
```
**Pros:** Low write latency, better performance
**Cons:** Risk of data loss, complex implementation
**Use Case:** Write-heavy applications

#### 4. Refresh-Ahead
```
Cache -> Proactively refresh before expiration
```
**Pros:** Reduced latency, always fresh data
**Cons:** Additional load, complexity
**Use Case:** Predictable access patterns

### Cache Levels

| Level | Examples | Latency | Capacity | Use Case |
|-------|----------|---------|----------|----------|
| **L1 - Browser** | Browser cache | ~0ms | MB | Static assets |
| **L2 - CDN** | CloudFlare, CloudFront | ~10ms | TB | Global content |
| **L3 - Load Balancer** | NGINX cache | ~1ms | GB | API responses |
| **L4 - Application** | In-memory cache | ~0.1ms | GB | Session data |
| **L5 - Database** | Query result cache | ~1ms | GB | Query results |

### Cache Eviction Policies

| Policy | Description | Use Case | Pros | Cons |
|--------|-------------|----------|------|------|
| **LRU** | Least Recently Used | General purpose | Good hit ratio | Overhead tracking |
| **LFU** | Least Frequently Used | Static content | Avoids cache pollution | Poor for changing patterns |
| **FIFO** | First In, First Out | Simple scenarios | Simple implementation | Ignores usage patterns |
| **TTL** | Time To Live | Time-sensitive data | Prevents stale data | May evict useful data |

---

## Microservices vs Monolith

### Detailed Comparison

| Aspect | Monolith | Microservices |
|--------|----------|---------------|
| **Architecture** | Single deployable unit | Multiple independent services |
| **Development** | Shared codebase | Separate codebases |
| **Deployment** | Deploy entire application | Deploy services independently |
| **Scaling** | Scale entire application | Scale individual services |
| **Technology Stack** | Single technology stack | Multiple technology stacks |
| **Data Management** | Shared database | Database per service |
| **Team Structure** | Single team | Multiple specialized teams |
| **Complexity** | Lower operational complexity | Higher operational complexity |

### When to Choose Monolith

**Advantages:**
- Simple deployment and testing
- Easy to debug and monitor
- Better performance (no network calls)
- ACID transactions across entire application
- Lower operational overhead

**Use Cases:**
- Small to medium applications
- Simple business domains
- Small development teams
- Prototypes and MVPs
- Applications with tight coupling requirements

### When to Choose Microservices

**Advantages:**
- Independent scaling and deployment
- Technology diversity
- Fault isolation
- Team autonomy
- Better for complex domains

**Use Cases:**
- Large, complex applications
- Multiple development teams
- Different scaling requirements per service
- Need for technology diversity
- High availability requirements

### Microservices Challenges & Solutions

| Challenge | Problem | Solution |
|-----------|---------|----------|
| **Service Discovery** | How services find each other | Service registry (Consul, Eureka) |
| **Configuration Management** | Managing configs across services | Centralized config server |
| **Distributed Transactions** | ACID across services | Saga pattern, eventual consistency |
| **Monitoring** | Tracking requests across services | Distributed tracing (Jaeger, Zipkin) |
| **Security** | Authentication/authorization | API Gateway, JWT tokens |
| **Data Consistency** | Maintaining consistency | Event sourcing, CQRS |

---

## Message Queues & Event Streaming

### Message Queue vs Event Streaming

| Aspect | Message Queue | Event Streaming |
|--------|---------------|-----------------|
| **Data Persistence** | Temporary (until consumed) | Persistent (configurable retention) |
| **Consumption Model** | Pull-based | Push/Pull hybrid |
| **Ordering** | FIFO within queue | Ordered within partition |
| **Scalability** | Vertical scaling | Horizontal scaling |
| **Use Case** | Task processing | Event-driven architecture |

### Message Queue Patterns

#### 1. Point-to-Point (Queue)
- One producer, one consumer
- Message consumed once
- Use case: Task processing, job queues

#### 2. Publish-Subscribe (Topic)
- One producer, multiple consumers
- Message delivered to all subscribers
- Use case: Notifications, event broadcasting

#### 3. Request-Reply
- Synchronous-like communication
- Correlation ID for matching
- Use case: RPC over messaging

### Popular Message Queue Technologies

| Technology | Type | Strengths | Weaknesses | Use Case |
|------------|------|-----------|------------|----------|
| **RabbitMQ** | Traditional MQ | Reliable, flexible routing | Complex setup | Complex routing needs |
| **Apache Kafka** | Event streaming | High throughput, durability | Complex, overkill for simple use cases | Event streaming, logs |
| **Amazon SQS** | Managed queue | Fully managed, scalable | Vendor lock-in | AWS-based applications |
| **Redis Pub/Sub** | In-memory | Fast, simple | No persistence | Real-time notifications |
| **Apache Pulsar** | Event streaming | Multi-tenancy, geo-replication | Newer, smaller ecosystem | Enterprise streaming |

### Event Streaming Patterns

#### 1. Event Sourcing
- Store events, not current state
- Reconstruct state from events
- Benefits: Audit trail, time travel, replay capability

#### 2. CQRS (Command Query Responsibility Segregation)
- Separate read and write models
- Optimized for different access patterns
- Benefits: Performance, scalability

#### 3. Saga Pattern
- Manage distributed transactions
- Choreography vs Orchestration
- Benefits: Microservices transactions

---

## API Design Patterns

### REST vs GraphQL vs gRPC

| Aspect | REST | GraphQL | gRPC |
|--------|------|---------|------|
| **Protocol** | HTTP | HTTP | HTTP/2 |
| **Data Format** | JSON, XML | JSON | Protocol Buffers |
| **Query Flexibility** | Fixed endpoints | Flexible queries | Strongly typed |
| **Caching** | HTTP caching | Complex caching | No built-in caching |
| **Learning Curve** | Easy | Moderate | Steep |
| **Performance** | Good | Variable | Excellent |

### REST API Design Principles

#### 1. Resource-Based URLs
```
Good: GET /users/123/orders
Bad:  GET /getUserOrders?userId=123
```

#### 2. HTTP Methods Usage
| Method | Purpose | Idempotent | Safe |
|--------|---------|------------|------|
| **GET** | Retrieve data | Yes | Yes |
| **POST** | Create resource | No | No |
| **PUT** | Update/replace resource | Yes | No |
| **PATCH** | Partial update | No | No |
| **DELETE** | Remove resource | Yes | No |

#### 3. Status Codes
| Code | Meaning | Use Case |
|------|---------|----------|
| **200** | OK | Successful GET, PUT, PATCH |
| **201** | Created | Successful POST |
| **204** | No Content | Successful DELETE |
| **400** | Bad Request | Invalid request data |
| **401** | Unauthorized | Authentication required |
| **403** | Forbidden | Access denied |
| **404** | Not Found | Resource doesn't exist |
| **500** | Internal Server Error | Server error |

### API Gateway Pattern

**Benefits:**
- Single entry point for all clients
- Cross-cutting concerns (auth, logging, rate limiting)
- Protocol translation
- Request/response transformation

**Features:**
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- Caching
- Monitoring and analytics

**Popular API Gateways:**
- AWS API Gateway
- Kong
- Zuul
- Ambassador
- Istio Gateway

---

## Content Delivery Networks (CDN)

### CDN Architecture

#### Push vs Pull CDNs

| Aspect | Push CDN | Pull CDN |
|--------|----------|----------|
| **Content Upload** | Manually upload to CDN | CDN fetches from origin |
| **Initial Setup** | More complex | Simple |
| **Storage Cost** | Pay for all uploaded content | Pay for cached content |
| **Content Updates** | Manual updates needed | Automatic updates |
| **Use Case** | Static content, large files | Dynamic content, frequent updates |

### CDN Strategies

#### 1. Geographic Distribution
- Edge servers close to users
- Reduced latency
- Better user experience

#### 2. Caching Strategies
- **Static Content**: Images, CSS, JavaScript
- **Dynamic Content**: API responses, personalized content
- **Streaming**: Video and audio content

#### 3. Cache Invalidation
- **TTL-based**: Time-based expiration
- **Event-based**: Invalidate on content changes
- **Purge**: Manual cache clearing

### CDN Providers Comparison

| Provider | Strengths | Weaknesses | Use Case |
|----------|-----------|------------|----------|
| **CloudFlare** | Security features, free tier | Less control over caching | General web applications |
| **AWS CloudFront** | AWS integration, global reach | Complex pricing | AWS-based applications |
| **Google Cloud CDN** | Google infrastructure | Smaller edge network | Google Cloud applications |
| **Fastly** | Real-time purging, VCL | Expensive | High-performance applications |

---

## Distributed Systems Concepts

### CAP Theorem

**Consistency, Availability, Partition Tolerance - Choose 2**

| Combination | Description | Examples | Trade-offs |
|-------------|-------------|----------|------------|
| **CA** | Consistent & Available | Traditional RDBMS | Not partition tolerant |
| **CP** | Consistent & Partition Tolerant | MongoDB, Redis | Sacrifice availability |
| **AP** | Available & Partition Tolerant | Cassandra, DynamoDB | Eventual consistency |

### ACID vs BASE Properties

| ACID | BASE |
|------|------|
| **Atomicity** | **Basically Available** |
| **Consistency** | **Soft State** |
| **Isolation** | **Eventual Consistency** |
| **Durability** | |

### Consensus Algorithms

#### 1. Raft Algorithm
- Leader election
- Log replication
- Safety guarantees
- Use case: Distributed databases

#### 2. Paxos Algorithm
- More complex than Raft
- Proven correctness
- Use case: Google's Chubby

#### 3. PBFT (Practical Byzantine Fault Tolerance)
- Handles malicious nodes
- Use case: Blockchain systems

### Distributed System Patterns

#### 1. Circuit Breaker
- Prevent cascading failures
- Fast failure detection
- Automatic recovery

#### 2. Bulkhead
- Isolate critical resources
- Prevent resource exhaustion
- Improve fault tolerance

#### 3. Retry with Backoff
- Handle transient failures
- Exponential backoff
- Jitter to prevent thundering herd

---

## System Monitoring & Observability

### The Three Pillars of Observability

#### 1. Metrics
- Quantitative measurements
- Time-series data
- Aggregated values
- Examples: CPU usage, request rate, error rate

#### 2. Logs
- Discrete events
- Structured or unstructured
- Searchable and filterable
- Examples: Error logs, access logs, application logs

#### 3. Traces
- Request flow through system
- Distributed tracing
- Performance analysis
- Examples: Request spans, service dependencies

### Monitoring Strategies

| Strategy | Purpose | Tools | Metrics |
|----------|---------|-------|---------|
| **Infrastructure** | Server health | Prometheus, Grafana | CPU, memory, disk, network |
| **Application** | App performance | New Relic, DataDog | Response time, throughput, errors |
| **Business** | KPIs | Custom dashboards | Revenue, user engagement, conversions |
| **Security** | Threat detection | SIEM tools | Failed logins, anomalies, intrusions |

### SLA vs SLO vs SLI

| Term | Definition | Example |
|------|------------|---------|
| **SLA** | Service Level Agreement | 99.9% uptime guarantee |
| **SLO** | Service Level Objective | 99.9% uptime target |
| **SLI** | Service Level Indicator | Actual uptime measurement |

### Error Budgets
- Acceptable amount of unreliability
- Based on SLO targets
- Balance between reliability and feature velocity

---

## Security Architecture

### Security Layers

#### 1. Network Security
- **Firewalls**: Control network traffic
- **VPNs**: Secure remote access
- **DDoS Protection**: Mitigate attacks
- **Network Segmentation**: Isolate systems

#### 2. Application Security
- **Authentication**: Identity verification
- **Authorization**: Access control
- **Input Validation**: Prevent injection attacks
- **Session Management**: Secure user sessions

#### 3. Data Security
- **Encryption at Rest**: Protect stored data
- **Encryption in Transit**: Protect data transfer
- **Data Masking**: Hide sensitive information
- **Key Management**: Secure cryptographic keys

### Authentication & Authorization

#### Authentication Methods

| Method | Description | Pros | Cons | Use Case |
|--------|-------------|------|------|----------|
| **Basic Auth** | Username/password in header | Simple | Not secure alone | Internal tools |
| **JWT** | JSON Web Token | Stateless, scalable | Token size, revocation | APIs, microservices |
| **OAuth 2.0** | Delegated authorization | Third-party integration | Complex | Social logins |
| **SAML** | Security assertion markup | Enterprise SSO | Complex setup | Enterprise applications |

#### Authorization Patterns

##### 1. Role-Based Access Control (RBAC)
- Users assigned to roles
- Roles have permissions
- Simple and widely adopted

##### 2. Attribute-Based Access Control (ABAC)
- Fine-grained permissions
- Context-aware decisions
- More flexible but complex

##### 3. Policy-Based Access Control
- Centralized policy management
- Declarative access rules
- Good for complex scenarios

### Common Security Threats & Mitigations

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **SQL Injection** | Malicious SQL code injection | Parameterized queries, input validation |
| **XSS** | Cross-site scripting | Input sanitization, CSP headers |
| **CSRF** | Cross-site request forgery | CSRF tokens, SameSite cookies |
| **Man-in-the-Middle** | Intercepting communications | HTTPS, certificate pinning |
| **DDoS** | Distributed denial of service | Rate limiting, CDN, DDoS protection |

---

## Common System Design Patterns

### Architectural Patterns

#### 1. Layered Architecture
- **Presentation Layer**: UI components
- **Business Layer**: Business logic
- **Data Layer**: Data access
- **Benefits**: Separation of concerns, testability
- **Drawbacks**: Performance overhead, coupling

#### 2. Hexagonal Architecture (Ports & Adapters)
- **Core**: Business logic
- **Ports**: Interfaces
- **Adapters**: External systems
- **Benefits**: Testability, flexibility
- **Drawbacks**: Complexity

#### 3. Event-Driven Architecture
- **Events**: State changes
- **Event Producers**: Generate events
- **Event Consumers**: Process events
- **Benefits**: Loose coupling, scalability
- **Drawbacks**: Complexity, debugging

### Design Patterns for Scalability

#### 1. Database Patterns

##### Read Replicas
- **Purpose**: Scale read operations
- **Implementation**: Master-slave replication
- **Benefits**: Improved read performance
- **Considerations**: Eventual consistency

##### CQRS (Command Query Responsibility Segregation)
- **Purpose**: Separate read/write operations
- **Implementation**: Different models for commands and queries
- **Benefits**: Optimized performance
- **Considerations**: Complexity, eventual consistency

##### Database Sharding
- **Purpose**: Distribute data across multiple databases
- **Implementation**: Horizontal partitioning
- **Benefits**: Scalability
- **Considerations**: Complexity, cross-shard operations

#### 2. Caching Patterns

##### Cache-Aside
- **Purpose**: Application manages cache
- **Implementation**: Check cache, fallback to database
- **Benefits**: Control over caching logic
- **Considerations**: Cache misses, stale data

##### Write-Through
- **Purpose**: Synchronous cache updates
- **Implementation**: Write to cache and database simultaneously
- **Benefits**: Consistency
- **Considerations**: Write latency

##### Write-Behind
- **Purpose**: Asynchronous cache updates
- **Implementation**: Write to cache first, database later
- **Benefits**: Performance
- **Considerations**: Data loss risk

#### 3. Communication Patterns

##### API Gateway
- **Purpose**: Single entry point for APIs
- **Implementation**: Proxy pattern
- **Benefits**: Centralized concerns
- **Considerations**: Single point of failure

##### Service Mesh
- **Purpose**: Service-to-service communication
- **Implementation**: Sidecar proxy pattern
- **Benefits**: Observability, security
- **Considerations**: Complexity, performance overhead

##### Event Sourcing
- **Purpose**: Store events instead of current state
- **Implementation**: Event store
- **Benefits**: Audit trail, flexibility
- **Considerations**: Complexity, eventual consistency

### Performance Optimization Patterns

#### 1. Database Optimization
- **Indexing**: Speed up queries
- **Query Optimization**: Efficient SQL
- **Connection Pooling**: Reuse connections
- **Batch Processing**: Reduce round trips

#### 2. Application Optimization
- **Lazy Loading**: Load data on demand
- **Eager Loading**: Load related data upfront
- **Pagination**: Limit result sets
- **Compression**: Reduce payload size

#### 3. Network Optimization
- **CDN**: Cache static content
- **HTTP/2**: Multiplexing, server push
- **Compression**: Gzip, Brotli
- **Keep-Alive**: Reuse connections

---

## Key Takeaways & Best Practices

### 1. Start Simple, Scale Gradually
- Begin with monolith, evolve to microservices
- Vertical scaling before horizontal
- Add complexity only when needed

### 2. Design for Failure
- Assume components will fail
- Implement circuit breakers
- Plan for graceful degradation

### 3. Monitor Everything
- Metrics, logs, and traces
- Set up alerts for critical issues
- Regular capacity planning

### 4. Security by Design
- Defense in depth
- Principle of least privilege
- Regular security reviews

### 5. Consider Trade-offs
- Consistency vs Availability
- Performance vs Complexity
- Cost vs Scalability

### 6. Documentation & Communication
- Document architectural decisions
- Share knowledge across teams
- Regular architecture reviews

---

## Interview Tips

### Common Questions to Prepare
1. **Design a URL shortener (bit.ly)**
2. **Design a chat system (WhatsApp)**
3. **Design a news feed (Twitter)**
4. **Design a video streaming service (YouTube)**
5. **Design a ride-sharing service (Uber)**

### Problem-Solving Approach
1. **Clarify Requirements**: Functional and non-functional
2. **Estimate Scale**: Users, data, requests
3. **High-Level Design**: Major components
4. **Detailed Design**: Deep dive into components
5. **Scale the Design**: Handle growth
6. **Address Bottlenecks**: Identify and solve issues

### Remember
- There's no single correct answer
- Focus on trade-offs and reasoning
- Ask clarifying questions
- Start simple, then add complexity
- Consider real-world constraints